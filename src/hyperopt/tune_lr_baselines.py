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
from data.dataset import IntrusionData

from baselines.model.reconstruction import AutoEncoder
from baselines.model.one_class import DeepSVDD

from baselines.trainer.reconstruction import AutoEncoderTrainer
from baselines.trainer.one_class import DeepSVDDTrainer

CLASS_SOLVER = {
    'AE': (AutoEncoder, AutoEncoderTrainer),
    'DeepSVDD': (DeepSVDD, DeepSVDDTrainer),
}

n_epochs = 200
num_samples = 3 # number of grid combinations to try

MODELS = CLASS_SOLVER.keys()
DATASETS = ['unswnb15', '5gnidd']

def train_func(
    config,
    path_to_dataset: str,
    model: str,
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

    dataset = IntrusionData(path_to_dataset)

    train_set, test_set = dataset.split_train_test(test_ratio=test_ratio, contamination_rate=contamination_rate, pos_label=anomaly_label)
    train_ldr = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_ldr = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    # define model
    model_class, trainer_class = CLASS_SOLVER[model]

    model = model_class(
        in_features=dataset.in_features,
        n_instances=dataset.n_instances,
        device='cuda',
        mlp_params=mlp_params,
        **model_params,
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

        trainer.evaluate(test_ldr)
        metrics = trainer.metric_values
        trainer.metric_values = {"Precision": [], "Recall": [], "F1-Score": [], "AUPR": [], "Accuracy": [], "AUROC": []}

        metrics = {metric: metrics[metric][-1] for metric in ['AUROC', 'F1-Score']}
        session.report({**metrics, **{'combined_metrics': sum(metrics.values())/len(metrics)}})

config_file = os.path.join('hyperopt', 'baselines', 'best_config_baselines.yml')
assert os.path.exists(config_file), 'Run tune_baselines.py before'

for dataset in DATASETS:
    for model in MODELS:

            # read best config
            with open(config_file) as f:
                best_configs = yaml.load(f, Loader=SafeLoader)


            curr_config = best_configs[dataset][model]

            search_space = {
                'lr': tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5])
            }

            # filter model params
            model_params = {key: val for key, val in curr_config.items() if key not in
                            ['lr']}

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
                name='_'.join(['Hyperopt_lr', dataset, model]),
                stop=TrialPlateauStopper(opt_metric, std=0.001, num_results=8, grace_period=8),
                log_to_file='trial.log'
            )

            tuner = tune.Tuner(
                tune.with_resources(
                    partial(
                        train_func,
                        path_to_dataset=path_to_dataset,
                        model = model,
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
            out_dir = os.path.join('hyperopt', 'baselines', 'lr', dataset)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            results.get_dataframe().to_csv(os.path.join(out_dir, model+'.csv')) # safe all results

            # safe best config
            best_configs[dataset][model]['lr'] = float(best_trial.config['lr'])
            with open(config_file, 'w') as f:
                yaml.dump(best_configs, f, sort_keys=True)
