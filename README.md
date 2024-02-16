==============================

Optimized hyperparameters can be found for the non-contrastive SSL models in [best_config.yml](hyperopt/best_config.yml),  and the optimized hyperparameters for the two baselines are stored in [best_config_baselines.yml](hyperopt/baselines/best_config_baselines.yml)

# Installation
```bash
git clone https://github.com/renje4z335jh4/non_contrastive_SSL_NIDS.git
conda create --name [env_name] python=3.8
conda activate [env_name]
cd non_contrastive_SSL_NIDS
pip install -e .
```

# Usage Guide
## Datasets

### 5G-NIDD
You can download the `Combined.zip` [here](https://etsin.fairdata.fi/dataset/9d13ef28-2ca7-44b0-9950-225359afac65/data)
Please extract the csv-file in 'data/raw/5GNIDD/'

### UNSW-NB15 (If the links get expired, check the official website of the dataset)
`UNSW_NB15_Test.csv`: [here](https://unsw-my.sharepoint.com/:x:/r/personal/z5025758_ad_unsw_edu_au/_layouts/15/Doc.aspx?sourcedoc=%7B2A810F6A-CC3D-4D98-909E-37489D8DAF98%7D&file=UNSW_NB15_testing-set.csv&action=default&mobileredirect=true)

`UNSW_NB15_Train.csv`: [here](https://unsw-my.sharepoint.com/:x:/r/personal/z5025758_ad_unsw_edu_au/_layouts/15/Doc.aspx?sourcedoc=%7B49413D38-3330-4358-BFA2-0349031198A5%7D&file=UNSW_NB15_training-set.csv&action=default&mobileredirect=true)

Please extract the csv-file in 'data/raw/UNSW-NB15/'

## Processing the Datasets
The [process script](./src/data/process.py) can be used as follows:
```bash
python src/data/process.py [data_set] -d [path/to/dir/containing/the/CSV/files] -o [path/to/output/dir]
```

### Example
```bash
python src/data/process.py UNSW-NB15 -d data/raw/UNSW-NB15/ -o data/processed/
```

## Reproduce the Experiments

### Hyperparameter tuning
To reproduce the experiments of the paper, we will start by executing the hyper-optimization. This can be simply done by executing:
```bash
src/hyperopt/full_hyperopt.sh
```
#### Details
The bash script will run three python files, which are hyper-optimizing each combination of model, encoder and augmentation on the data sets *UNSW-NB15* and *5G-NIDD*, where the models are *BarlowTwins*, *BYOL*, *SimSiam*, *VICReg* and *W-MSE*; the encoders are a CNN, a MLP and a FT-Transformer; and the augmentations are *Gaussian Noise*, *Mixup*, *Random Shuffle*, *Subsets*, *SwapNoise* (also called *CutMix*) and *Zero Out Noise*. In total 5 * 6 * 3 = 90 different experiments are tuned for the two datasets.

Step by step the script is optimizing:
1. Model and augmentation parameters
2. Learning rate
3. Number of epochs to train the model

Similarly, the baselines *Deep AutoEncoder* and *DeepSVDD* are tuned with
```bash
src/hyperopt/full_hyperopt_baselines.sh
```

Note: This could take a significant amount of time and resources!

### Model Training
The hyper-parameters collected in the tuning step are stored in a [YAML file](hyperopt/best_config.yml). The final results are gathered by training the different models with the best hyper-parameters for 10 runs and averaging the performance metrics. This is done by executing
```bash
python src/utils/run_models.py
```
The result of each model together with the weights of the best performing run (depicted by the *AUROC*) is stored under `run_results/`.

In a similar way, results of the optimized baselines are generated by executing
```bash
python src/utils/run_baselines.py
```



## Useful information about the trainer parameter
1. Parameters like model, batch_size, n_epochs, device are self-explaining
2. anomaly_label parameter indicates which label is the malicious label (commonly 1)
3. ckpt_root - if given a checkpoint, will be saved every 5 epochs. Set to None, if you only want to save the model regularly
4. safe_best_model - if given the trainer, will save the best model (lowest loss) as checkpoint.
