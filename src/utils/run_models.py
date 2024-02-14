from collections import defaultdict
from datetime import datetime
from typing import Dict
import yaml
from yaml.loader import SafeLoader
import os
import numpy as np
from torch.utils.data import DataLoader

# imports
from data.dataset import MultiViewIntrusionData, MultiViewIntrusionDataTransformer
from data.augmentation import MultiViewDataInjector
from utils.metrics import average_results

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

N_RUNS = 10
RESULTS_PATH = 'run_results'

MODELS = CLASS_SOLVER.keys()
ENCODERS = ['CNN', 'MLP_Encoder', 'FTTransformerEncoder']
DATASETS = ['unswnb15', '5gnidd' ]

GENERAL_PARAMS = {
    'weight_decay': 1e-4,
    'test_ratio': 0.5,
    'contamination_rate': 0.0,
    'anomaly_label': 1,
    'batch_size': 1024,
    'mlp_params': {
        'embedding_dim': 256,
        'output_dim': 256,
        'n_layers': 2,
        'batch_norm': True,
        'dropout': False,
    }
}

def train_func(
    path_to_dataset: str,
    encoder_class: str,
    model_name: str,
    augmentation_params: Dict,
    model_params: Dict,
    lr: float,
    n_epochs: int,
    n_runs: int,
    dataset_name: str,
    augmentation_name: str,
) -> np.ndarray:

    ####### fixed params -> GENERAL_PARAMS
    drop_last_batch = model_name == 'WMSE'

    #  define augmentations
    train_transform = MultiViewDataInjector(augmentation_params['transformations'], augmentation_params['n_subsets'], augmentation_params['overlap'], training=True)
    test_transform = None if augmentation_params['subsets'] is False else MultiViewDataInjector([None] * augmentation_params['n_subsets'], augmentation_params['n_subsets'], augmentation_params['overlap'], training=False)

    loss_values, time_values = [], []
    metric_values = defaultdict(list)
    best_auroc = -1
    for i_run in range(n_runs):
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

        train_set, test_set = dataset.split_train_test(test_ratio=GENERAL_PARAMS['test_ratio'], contamination_rate=GENERAL_PARAMS['contamination_rate'], pos_label=GENERAL_PARAMS['anomaly_label'])
        train_ldr = DataLoader(dataset=train_set, batch_size=GENERAL_PARAMS['batch_size'], shuffle=True, drop_last=drop_last_batch)
        test_ldr = DataLoader(dataset=test_set, batch_size=GENERAL_PARAMS['batch_size'], shuffle=False)

        # define model
        model_class, trainer_class = CLASS_SOLVER[model_name]

        if model_name == 'VICReg' or model_name == 'BarlowTwins':
            model_params['batch_size'] = GENERAL_PARAMS['batch_size']

        model = model_class(
            in_features=dataset.in_features,
            n_instances=dataset.n_instances,
            device='cuda',
            encoder_class=encoder_class,
            mlp_params=GENERAL_PARAMS['mlp_params'],
            **model_params,
            **encoder_args
        )

        trainer = trainer_class(
            model=model,
            batch_size=GENERAL_PARAMS['batch_size'],
            lr=lr,
            weight_decay=GENERAL_PARAMS['weight_decay'],
            n_epochs=n_epochs,
            device='cuda',
            anomaly_label=GENERAL_PARAMS['anomaly_label'],
            test_ldr=None,
            ckpt_root=None,
        )

        # train model
        trainer.train(train_ldr)

        # store losses of the current run
        loss_values.append(trainer.loss_values)
        time_values.append(trainer.time_values)

        # eval model
        if subsets:
            train_set.transform.transformations = [None] * augmentation_params['n_subsets']
        else:
            train_set.transform = None

        kmeans = KMeans_Eval(
            encoder = model.get_encoder(),
            batch_size=GENERAL_PARAMS['batch_size'],
            device='cuda'
        )

        kmeans.fit(train_ldr)
        curr_metric_vals = kmeans.validate(test_ldr)

        for metric_name, metric_vals in curr_metric_vals.items():
            metric_values[metric_name].append(metric_vals)

        if curr_metric_vals['AUROC'] > best_auroc:
            best_auroc = curr_metric_vals['AUROC']
            trainer.save_ckpt(os.path.join(RESULTS_PATH, '_'.join([dataset_name, model_name, encoder_class, augmentation_name, 'weights.pt'])))
            print(f'Best model on run {i_run+1} with AUROC {best_auroc}')

    return metric_values, loss_values, time_values

config_file = os.path.join('hyperopt', 'best_config.yml')
assert os.path.exists(config_file), 'Run full_hyperopt.sh before'
os.makedirs(RESULTS_PATH, exist_ok=True)

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

                # filter model params
                model_params = {key: val for key, val in curr_config.items() if key not in
                                ['n_subsets', 'overlap', 'p', 'mean', 'std', 'lr', 'n_epochs']}


                path_to_dataset = 'data/processed/' + dataset + '.csv'

                metric_values, loss_values, time_values = train_func(
                    path_to_dataset=path_to_dataset,
                    encoder_class=encoder_class,
                    model_name=model,
                    augmentation_params=augmentation_params,
                    model_params=model_params,
                    lr=curr_config['lr'],
                    n_epochs=curr_config['n_epochs'],
                    n_runs=N_RUNS,
                    dataset_name=dataset,
                    augmentation_name=augmentation,
                )

                # Postprocessing
                avg_results = average_results({'loss': loss_values, 'time': time_values, **metric_values})

                print(dataset, model, encoder_class, augmentation, 'Results: ', avg_results)

                fname = os.path.join(RESULTS_PATH, '_'.join([dataset, model, encoder_class, augmentation, 'results.yml']))

                date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                with open(fname, 'w') as file:
                    yaml.dump(dict(sorted({
                        'time': date_time,
                        'model': model,
                        'data set': dataset,
                        'encoder': encoder_class,
                        'n_epochs': curr_config['n_epochs'],
                        'n_runs': N_RUNS,
                        **{'general_params': GENERAL_PARAMS},
                        **{'model_params': model_params},
                        **{'augmentation': augmentation_params},
                        **avg_results
                    }.items())), file)
