from collections import defaultdict
from datetime import datetime
from typing import Dict
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import os
import numpy as np
from torch.utils.data import DataLoader

# imports
from utils.metrics import average_results

from data.dataset import IntrusionData

from baselines.model.reconstruction import AutoEncoder
from baselines.model.one_class import DeepSVDD

from baselines.trainer.reconstruction import AutoEncoderTrainer
from baselines.trainer.one_class import DeepSVDDTrainer

CLASS_SOLVER = {
    'AE': (AutoEncoder, AutoEncoderTrainer),
    'DeepSVDD': (DeepSVDD, DeepSVDDTrainer),
}
N_RUNS = 10
RESULTS_PATH = 'run_results'

DATASETS = ['unswnb15', '5gnidd']
MODELS = CLASS_SOLVER.keys()

GENERAL_PARAMS = {
    'weight_decay': 1e-4,
    'test_ratio': 0.5,
    'contamination_rate': 0.0,
    'anomaly_label': 1,
    'batch_size': 1024,
}

def train_func(
    path_to_dataset: str,
    model_name: str,
    model_params: Dict,
    lr: float,
    n_epochs: int,
    n_runs: int,
    dataset_name: str,
) -> np.ndarray:

    ####### fixed params -> GENERAL_PARAMS

    loss_values, time_values = [], []
    metric_values = defaultdict(list)
    best_auroc = -1
    for i_run in range(n_runs):
        # load data set
        dataset = IntrusionData(path_to_dataset)

        train_set, test_set = dataset.split_train_test(test_ratio=GENERAL_PARAMS['test_ratio'], contamination_rate=GENERAL_PARAMS['contamination_rate'], pos_label=GENERAL_PARAMS['anomaly_label'])
        train_ldr = DataLoader(dataset=train_set, batch_size=GENERAL_PARAMS['batch_size'], shuffle=True)
        test_ldr = DataLoader(dataset=test_set, batch_size=GENERAL_PARAMS['batch_size'], shuffle=False)

        # define model
        model_class, trainer_class = CLASS_SOLVER[model_name]

        model = model_class(
            in_features=dataset.in_features,
            n_instances=dataset.n_instances,
            device='cuda',
            **model_params,
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
        trainer.evaluate(test_ldr)
        curr_metric_vals = trainer.metric_values

        for metric_name, metric_vals in curr_metric_vals.items():
            metric_values[metric_name].append(metric_vals[-1])

        if curr_metric_vals['AUROC'][-1] > best_auroc:
            best_auroc = curr_metric_vals['AUROC']
            trainer.save_ckpt(os.path.join(RESULTS_PATH, '_'.join([dataset_name, model_name, 'weights.pt'])))
            print(f'Best model on run {i_run+1} with AUROC {best_auroc}')

    return metric_values, loss_values, time_values

config_file = os.path.join('hyperopt', 'baselines', 'best_config_baselines.yml')
assert os.path.exists(config_file), 'Run full_hyperopt_baselines.sh before'
os.makedirs(RESULTS_PATH, exist_ok=True)

for dataset in DATASETS:
    for model in MODELS:
        # read best config
        with open(config_file) as f:
            best_configs = yaml.load(f, Loader=SafeLoader)

            curr_config = best_configs[dataset][model]

            # filter model params
            model_params = {key: val for key, val in curr_config.items() if key not in
                            ['lr', 'n_epochs']}

            path_to_dataset = str(Path(os.path.join(Path(__file__).parent.resolve(), '..', '..', 'data', 'processed', dataset + '.csv')).resolve())

            metric_values, loss_values, time_values = train_func(
                path_to_dataset=path_to_dataset,
                model_name=model,
                model_params=model_params,
                lr=curr_config['lr'],
                n_epochs=curr_config['n_epochs'],
                n_runs=N_RUNS,
                dataset_name=dataset,
            )

            # Postprocessing
            avg_results = average_results({'loss': loss_values, 'time': time_values, **metric_values})

            print(dataset, model, 'Results: ', avg_results)

            fname = os.path.join(RESULTS_PATH, '_'.join([dataset, model, 'results.yml']))

            date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            with open(fname, 'w') as file:
                yaml.dump(dict(sorted({
                    'time': date_time,
                    'model': model,
                    'data set': dataset,
                    'n_epochs': curr_config['n_epochs'],
                    'n_runs': N_RUNS,
                    **{'general_params': GENERAL_PARAMS},
                    **{'model_params': model_params},
                    **avg_results
                }.items())), file)
