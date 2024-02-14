from typing import Dict
from ray import tune
from ray.air import session
from ray.air import RunConfig
from ray.tune.stopper import TrialPlateauStopper
from pathlib import Path
from functools import partial
from ray.tune.schedulers import ASHAScheduler
import yaml
from yaml.loader import SafeLoader
import os
from torch.utils.data import DataLoader

# imports
from data.dataset import MultiViewIntrusionData, MultiViewIntrusionDataTransformer
from data.augmentation import MultiViewDataInjector

from models.ssl.byol import BYOL
from models.ssl.vicreg import VICReg
from models.ssl.barlowtwins import BarlowTwins
from models.ssl.simsiam import SimSiam
from models.ssl.wmse import WMSE

from models.ssl_evaluation.kmeans import KMeans_Eval

from trainer.ssl.byol import BYOL_Trainer
from trainer.ssl.vicreg import VICReg_Trainer
from trainer.ssl.barlowtwins import BarlowTwins_Trainer
from trainer.ssl.simsiam import SimSiam_Trainer
from trainer.ssl.wmse import WMSE_Trainer

CLASS_SOLVER = {
    'BYOL': (BYOL, BYOL_Trainer),
    'VICReg': (VICReg, VICReg_Trainer),
    'BarlowTwins': (BarlowTwins, BarlowTwins_Trainer),
    'SimSiam': (SimSiam, SimSiam_Trainer),
    'WMSE': (WMSE, WMSE_Trainer),
}

n_epochs = 200
num_samples = 3 # number of grid combinations to try

MODELS = CLASS_SOLVER.keys()
ENCODERS = ['CNN', 'MLP_Encoder', 'FTTransformerEncoder']
DATASETS = ['unswnb15', '5gnidd']

def train_func(
    config,
    path_to_dataset: str,
    encoder_class: str,
    model: str,
    augmentation_params: Dict,
    model_params: Dict,
    n_epochs: int = 10,
):

    ####### fixed params
    weight_decay = 1e-4
    test_ratio = 0.5
    contamination_rate = 0.0
    anomaly_label = 1
    batch_size = 1024
    mlp_params = {
        'embedding_dim': 256,
        'output_dim': 256,
        'n_layers': 2,
        'batch_norm': True,
        'dropout': False,
    }
    #######
    drop_last_batch = model == 'WMSE'

    #  define augmentations
    train_transform = MultiViewDataInjector(augmentation_params['transformations'], augmentation_params['n_subsets'], augmentation_params['overlap'], training=True)
    test_transform = None if augmentation_params['subsets'] is False else MultiViewDataInjector([None] * augmentation_params['n_subsets'], augmentation_params['n_subsets'], augmentation_params['overlap'], training=False)

    # load data set
    encoder_args = {}
    if encoder_class == 'FTTransformerEncoder':
        dataset = MultiViewIntrusionDataTransformer(path_to_dataset, train_transform, test_transform, shuffle_features=subsets)
        encoder_args = {
            'categorical_col_indices': dataset.categorical_cols_idx,
            'categories_unique_values': dataset.unique_cats,
            'numeric_col_indices': dataset.numeric_cols_idx
        }
        if augmentation_params['n_subsets'] > 2 or augmentation_params['overlap'] < 1.0:
            # workaround
            encoder_args['numeric_col_indices'] = range(0, (train_transform.n_overlap + train_transform.n_features_subset))
    else:
        dataset = MultiViewIntrusionData(path_to_dataset, train_transform, test_transform, shuffle_features=subsets)
        encoder_args = {'num_features': dataset.in_features}

    train_set, test_set = dataset.split_train_test(test_ratio=test_ratio, contamination_rate=contamination_rate, pos_label=anomaly_label)
    train_ldr = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=drop_last_batch)
    test_ldr = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    # define model
    model_class, trainer_class = CLASS_SOLVER[model]

    if model == 'VICReg' or model == 'BarlowTwins':
        model_params['batch_size'] = batch_size

    model = model_class(
        in_features=dataset.in_features,
        n_instances=dataset.n_instances,
        device='cuda',
        encoder_class=encoder_class,
        mlp_params=mlp_params,
        **model_params,
        **encoder_args
    )

    trainer = trainer_class(
        model=model,
        batch_size=batch_size,
        lr=config['lr'],
        weight_decay=weight_decay,
        n_epochs=1, # increased later
        device='cuda',
        anomaly_label=anomaly_label,
        test_ldr=None,
        ckpt_root=None,
    )

    for epoch in range(n_epochs):

        trainer.n_epochs = epoch+1
        trainer.train(train_ldr)

        transform_copy = train_set.transform
        if subsets:
            train_set.transform.transformations = [None] * augmentation_params['n_subsets']
        else:
            train_set.transform = None

        kmeans = KMeans_Eval(
            encoder = model.get_encoder(),
            batch_size=batch_size,
            device='cuda'
        )

        kmeans.fit(train_ldr)
        metrics = kmeans.validate(test_ldr)

        train_set.transform = transform_copy
        metrics = {metric: metrics[metric] for metric in ['AUROC', 'F1-Score']}
        session.report({**metrics, **{'combined_metrics': sum(metrics.values())/len(metrics)}})

config_file = os.path.join('hyperopt', 'best_config.yml')
assert os.path.exists(config_file), 'Run tune_epochs.py before'

for dataset in DATASETS:
    for model in MODELS:
        for encoder_class in ENCODERS:

            # read best config
            with open(config_file) as f:
                best_configs = yaml.load(f, Loader=SafeLoader)

            for augmentation in best_configs[dataset][model][encoder_class].keys():

                curr_config = best_configs[dataset][model][encoder_class][augmentation]

                # define augmentation
                if augmentation == 'subsets':
                    subsets = True
                    n_subsets = curr_config['n_subsets']
                    overlap = curr_config['overlap']
                    transformations = [None]*n_subsets
                else:
                    n_subsets = 2
                    overlap = 1.0
                    subsets = False

                    if augmentation == 'mixup':
                        mixup_alpha = curr_config['mixup_alpha']
                        transformations = [None]*n_subsets
                    elif augmentation == 'randomshuffle':
                        transformations = [
                            [{'FisherYates':{}}],
                            [{'FisherYates':{}}],
                        ]
                    elif augmentation == 'gaussiannoise':
                        transformations = [
                            [{'GaussianNoise': {'p': curr_config['p'], 'mean': curr_config['mean'], 'std': curr_config['std']}}],
                            [{'GaussianNoise': {'p': curr_config['p'], 'mean': curr_config['mean'], 'std': curr_config['std']}}],
                        ]
                    else:
                        transformations = [
                            [{augmentation: {'p': curr_config['p']}}],
                            [{augmentation: {'p': curr_config['p']}}],
                        ]

                augmentation_params = {
                    'subsets': subsets,
                    'n_subsets': n_subsets,
                    'overlap': overlap,
                    'transformations': transformations

                }

                search_space = {
                    'lr': tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5])
                }

                # filter model params
                model_params = {key: val for key, val in curr_config.items() if key not in
                                ['n_subsets', 'overlap', 'p', 'mean', 'std', 'lr']}

                path_to_dataset = str(Path(os.path.join(Path(__file__).parent.resolve(), '..', '..', 'data', 'processed', dataset + '.csv')).resolve())

                scheduler = ASHAScheduler()
                opt_metric = 'combined_metrics'
                tune_config = tune.TuneConfig(
                    metric=opt_metric,
                    mode='max',
                    num_samples=num_samples,
                    scheduler=scheduler,
                    search_alg=None,
                )

                tune_run_config = RunConfig(
                    name='_'.join(['Hyperopt_lr', dataset, model, encoder_class, augmentation]),
                    stop=TrialPlateauStopper(opt_metric, std=0.001, num_results=8, grace_period=8),
                    log_to_file='trial.log'
                )

                tuner = tune.Tuner(
                    tune.with_resources(
                        partial(
                            train_func,
                            path_to_dataset=path_to_dataset,
                            encoder_class = encoder_class,
                            model = model,
                            augmentation_params = augmentation_params,
                            n_epochs = n_epochs,
                            model_params = model_params
                        ),
                        {'gpu': 1}
                    ),
                    param_space=search_space,
                    tune_config=tune_config,
                    run_config=tune_run_config
                )

                results = tuner.fit()


                best_trial = results.get_best_result(opt_metric, "max", scope="last")
                print(f"Best trial config: {best_trial.config}")
                print(f"Best trial final F1: {best_trial.metrics_dataframe['F1-Score'].values[-1]}")
                print(f"Best trial final AUROC: {best_trial.metrics_dataframe['AUROC'].values[-1]}")
                print(f"Best trial final combined metrics: {best_trial.metrics_dataframe['combined_metrics'].values[-1]}")

                # store results
                out_dir = os.path.join('hyperopt', 'lr', dataset, model, encoder_class)
                Path(out_dir).mkdir(parents=True, exist_ok=True)

                results.get_dataframe().to_csv(os.path.join(out_dir, augmentation+'.csv')) # safe all results

                # safe best config
                best_configs[dataset][model][encoder_class][augmentation]['lr'] = float(best_trial.config['lr'])
                with open(config_file, 'w') as f:
                    yaml.dump(best_configs, f, sort_keys=True)
