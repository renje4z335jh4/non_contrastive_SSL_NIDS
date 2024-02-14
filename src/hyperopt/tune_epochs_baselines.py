from collections import defaultdict
from typing import Dict
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import os
import numpy as np
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
n_runs = 3

MODELS = CLASS_SOLVER.keys()
DATASETS = ['unswnb15', '5gnidd']
OPT_METRICS = ['AUROC', 'F1-Score']

def train_func(
    path_to_dataset: str,
    model_name: str,
    model_params: Dict,
    lr: float,
    n_epochs: int = 10,
    n_runs: int = 3
) -> np.ndarray:

    ####### fixed params
    weight_decay = 1e-4
    test_ratio = 0.5
    contamination_rate = 0.0
    anomaly_label = 1
    batch_size = 1024
    #######

    metric_values = defaultdict(list)
    for i_run in range(n_runs):
        # load data set
        dataset = IntrusionData(path_to_dataset)

        train_set, test_set = dataset.split_train_test(test_ratio=test_ratio, contamination_rate=contamination_rate, pos_label=anomaly_label)
        train_ldr = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        test_ldr = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

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
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            n_epochs=1, # increased later
            device='cuda',
            anomaly_label=anomaly_label,
            test_ldr=None,
            ckpt_root=None,
        )

        # evaluate after 5 * n epochs training
        for epoch in range(5, n_epochs+1, 5):

            trainer.n_epochs = epoch
            trainer.train(train_ldr)

            trainer.evaluate(test_ldr)
            curr_metric_vals = trainer.metric_values
            trainer.metric_values = {"Precision": [], "Recall": [], "F1-Score": [], "AUPR": [], "Accuracy": [], "AUROC": []}

            metric_values[i_run].append(sum([curr_metric_vals[m][-1] for m in OPT_METRICS])/len(OPT_METRICS))

    # average
    return np.asarray([metric_values[i_run] for i_run in range(n_runs)]).mean(axis=0)

config_file = os.path.join('hyperopt', 'baselines', 'best_config_baselines.yml')
assert os.path.exists(config_file), 'Run tune_baselines.py before'

for dataset in DATASETS:
    for model in MODELS:

            # read best config
            with open(config_file) as f:
                best_configs = yaml.load(f, Loader=SafeLoader)

            curr_config = best_configs[dataset][model]

            # filter model params
            model_params = {key: val for key, val in curr_config.items() if key not in
                            ['lr']}

            path_to_dataset = str(Path(os.path.join(Path(__file__).parent.resolve(), '..', '..', 'data', 'processed', dataset + '.csv')).resolve())

            avg_metric_epochs = train_func(
                path_to_dataset=path_to_dataset,
                model_name=model,
                model_params=model_params,
                lr=curr_config['lr'],
                n_epochs=n_epochs,
                n_runs=n_runs
            )

            best_index = np.argmax(avg_metric_epochs)
            best_n_epochs = (best_index + 1) * 5
            print(f"Best n_epochs: {best_n_epochs}")
            print(f"Best trial final average of {', '.join(OPT_METRICS)}: {avg_metric_epochs[best_index]}")

            # store results
            out_dir = os.path.join('hyperopt', 'baselines', 'epochs', dataset)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            np.save(os.path.join(out_dir, model+'.npy'), avg_metric_epochs) # safe all results

            # safe best config
            best_configs[dataset][model]['n_epochs'] = int(best_n_epochs)
            with open(config_file, 'w') as f:
                yaml.dump(best_configs, f, sort_keys=True)
