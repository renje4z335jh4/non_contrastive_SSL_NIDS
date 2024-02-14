import argparse
import os
from typing import Dict
import logging
import math
import pandas as pd
from tqdm import tqdm
import numpy as np
import yaml

from data.helper import get_files, get_file_len, min_max_normalization, one_hot_encoding, get_min_dtype
logging.basicConfig(level=logging.INFO)

CHUNK_SIZE = 100000
THRESHOLD_CATS = 150
THRESHOLD_NAN = 0.2
TMP_FILE = 'tmp/tmp_processing.csv'

STATS = {
    'n_samples': 0,
    'n_dropped_samples': 0,
    'n_dropped_features': 0,
    'n_duplicates': 0
}

def prepare_col_info(file: str, config) -> Dict[str, dict]:
    """Builds structure for later processing

    Parameters
    ----------
    file : str
        single file to collect initial information

    Returns
    -------
    Dict[str, dict]
        information/default values for each column:
        dtype -> may shrink dtype: np.min_scalar_type
        drop -> True/False, for whether column should be dropped (timestamp, addresses)
        min -> maximum value of corresponding dtype
        max -> minimum value of corresponding dtype
        uniques -> list of unique values for categorical features
    """

    # take column names from config if exists
    # else take column names from file
    if not config.get('column_names'):
        kwargs = {'header': 0}
    else:
        kwargs = {
            'header': None,
            'names': config.get('column_names')
        }

    kwargs['index_col'] = 0 if config.get('has_index_col') else None

    # read subset
    small_df = pd.read_csv(file, nrows=10000, skipinitialspace=True, **kwargs)
    # drop misleading NaN values
    small_df.dropna(axis=0, inplace=True)

    col_info = {}
    for col in small_df.columns:
        dtype=small_df[col].dtype
        col_info[col] = {
            'dtype': dtype,
            'drop': False,
            'num_nan': 0,
        }

        if config.get('cols2remove') and col in config.get('cols2remove'):
            col_info[col]['drop'] = True # mark column to be dropped, when marked as to drop in config

        if col in small_df.select_dtypes(include=np.number).columns:
            # numerical columns
            if np.issubdtype(dtype, np.floating):
                info = np.finfo(dtype) # float
            else:
                info = np.iinfo(dtype) # integer
            col_info[col]['min'] = info.max
            col_info[col]['max'] = info.min

        else:
            # categorical columns
            col_info[col]['uniques'] = []
    return col_info


def first_iteration(files: list, col_info: Dict[str, dict], config) -> Dict[str, dict]:
    """ Goes through each file by chunks (limit RAM usage):
    -> collect min/max for each numeric column
    -> collect unique values for categorical features
    -> encode labels to 0/1
    -> save as one single temporary file
    -> rename column names - strip, lower, ' ' to '_'

    Parameters
    ----------
    files : list
        list of files to process
    col_info : Dict[str, dict]
        collected column information

    Returns
    -------
    Dict[str, dict]
        update column information
    """

    cols_to_drop = [col for col, dict in col_info.items() if dict['drop'] is True]
    if len(cols_to_drop)>0:
        logging.info('Following columns will be dropped: %s', str(cols_to_drop))
        STATS['n_dropped_features'] += len(cols_to_drop)

    write_header = True
    # delete tmp file if exists
    if os.path.exists(TMP_FILE):
        os.remove(TMP_FILE)

    # take column names from config if exists
    # else take column names from file
    if not config.get('column_names'):
        kwargs = {'header': 0}
    else:
        kwargs = {
            'header': None,
            'names': config.get('column_names')
        }

    logging.info('Start first iteration over the data set')

    # do not read columns, we will drop anyway
    use_cols = [col for col, dict in col_info.items() if not dict['drop'] is True]

    # iterate over each file
    for file_num, file in enumerate(files):

        logging.info('Processing file %i of %i', file_num+1, len(files))

        for chunk in tqdm(pd.read_csv(file, chunksize=CHUNK_SIZE, iterator=True, skipinitialspace=True, usecols=use_cols, **kwargs), total=math.ceil(get_file_len(file)/CHUNK_SIZE)):
            chunk = chunk.copy()
            STATS['n_samples'] += len(chunk)

            # in CSE 2018, drop header repetition
            len_before = len(chunk)
            chunk = chunk[chunk[config.get('label_name')] != "Label"]
            if len(chunk) < len_before:
                logging.info('Dropped %i duplicated header rows in file %s', len_before - len(chunk), file)
                # header repetitions found, all dtypes are object
                for col in [col_name for col_name, dict in col_info.items() if np.issubdtype(dict['dtype'], np.number)]:
                    chunk[col] = pd.to_numeric(chunk[col])

            # find additional columns not available in each file
            remove_cols = list(chunk.columns[~chunk.columns.isin(col_info.keys())])
            if len(remove_cols) > 0: #
                # relevant for CSE
                logging.info('File %s has following additional columns which will be dropped: %s', file, str(remove_cols))
                # drop additional columns
                chunk.drop(remove_cols, axis=1, inplace=True)

            # check if dtype still matches
            # for the case that e.g. the first 10000 lines are integers but than the values are floats
            for col, dtype in chunk.dtypes.items():
                if col_info[col]['dtype'] == 'int64' and dtype != col_info[col]['dtype'] and dtype == 'float64':
                    col_info[col]['dtype'] = dtype

            # save number NaN values for each feature
            for feature, nans in chunk.isna().sum().to_dict().items():
                col_info[feature]['num_nan'] += nans

            # save min and max values of each column
            local_min = chunk.select_dtypes(np.number).min(axis=0).to_dict()
            local_max = chunk.select_dtypes(np.number).max(axis=0).to_dict()
            for feature in chunk.columns:
                if np.issubdtype(col_info[feature]['dtype'], np.number):
                    col_info[feature]['min'] = min(col_info[feature]['min'], local_min[feature])
                    col_info[feature]['max'] = max(col_info[feature]['max'], local_max[feature])
                else:
                    # save unique values for string columns
                    # if already more than THRESHOLD_CATS -> don't check it will be dropped anyway
                    col_info[feature]['uniques'] = np.unique(list(chunk[feature]) + list(col_info[feature]['uniques'])) if len(col_info[feature]['uniques']) <= THRESHOLD_CATS else col_info[feature]['uniques']

            # encode labels
            chunk[config.get('label_name')] = chunk[config.get('label_name')].apply(lambda label: 0 if str(label).lower() in ['normal', 'benign', '0'] else 1)

            # change name of config.get('label_name') column to 'label'
            chunk.rename(columns={config.get('label_name'): 'label'}, inplace=True)
            chunk.columns = [col_name.strip().lower().replace(' ', '_') for col_name in chunk.columns]

            # append to single tmp file
            chunk.to_csv(TMP_FILE, header=write_header, mode='a', index=False)

            write_header=False

    # after iteration label column only contains 0 and 1
    col_info[config.get('label_name')]['dtype'] = np.int8
    col_info[config.get('label_name')]['min'] = 0
    col_info[config.get('label_name')]['max'] = 1

    # remove deleted columns also from col_info
    for col in cols_to_drop:
        del col_info[col]

    # rename keys
    col_info['label'] = col_info.pop(config.get('label_name'))
    columns = list(col_info.keys())
    for key in columns:
        col_info[key.strip().lower().replace(' ', '_')] = col_info.pop(key)
    return col_info



def second_iteration(col_info: Dict[str, dict], output_file: str) -> None:
    """Iterates the tmp file
    -> shrink data type to minimum necessary
    -> replace inf/-inf with NaN
    -> drop static columns
    -> drop columns with more than THRESHOLD_NAN NaN values
    -> drop categorical columns with more than THRESHOLD_CATS unique values - no one hot encoding wanted/possible
    -> drop each row with NaN values
    -> apply min-max normalization with collected min/max values for each column
    -> encode categorical features
    -> drop duplicated rows
    -> delete tmp file
    -> save to new file

    Parameters
    ----------
    col_info : Dict[str, dict]
        column information
    output_file : str
        file to save the processed data set
    """

    if os.path.exists(output_file):
        os.remove(output_file)

    # drop cats (IPs etc)
    drop_cats = [feature for feature, dict in col_info.items() if (dict.get('uniques') is not None and (len(dict['uniques']) > THRESHOLD_CATS))]
    logging.info('Features %s will be dropped, because they consists of more than %i unique categorical features.', ', '.join(drop_cats), THRESHOLD_CATS)

    # find static columns
    static_cols = [feature for feature in col_info.keys() if (np.issubdtype(col_info[feature]['dtype'], np.number) and col_info[feature]['min'] == col_info[feature]['max'])]
    logging.info('Features %s are static and will be dropped.', str(static_cols))

    # drop features with more than THRESHOLD_NAN ration NaN values
    drop_nan_cols = [feature for feature, dict in col_info.items() if dict['num_nan']/STATS['n_samples'] > THRESHOLD_NAN]
    logging.info('Features %s will be dropped, because they consists of more than %i NaN values.', str(drop_nan_cols), int(THRESHOLD_NAN*100))

    drop = np.unique(np.asarray(drop_cats + static_cols + drop_nan_cols))
    STATS['n_dropped_features'] += len(drop)

    # shrink dtypes
    for feature in list(col_info.keys()):
        if ~np.issubdtype(col_info[feature]['dtype'], np.number):
            continue
        col_info[feature]['dtype'] = get_min_dtype(col_info[feature]['min'], col_info[feature]['max'])

    # set dtypes for each column
    dtypes = {feature: dict['dtype'] for feature, dict in col_info.items()}
    use_cols = [col_name for col_name in list(set(col_info.keys()) - set(drop))]
    all_chunks = []
    for chunk in tqdm(pd.read_csv(TMP_FILE, chunksize=CHUNK_SIZE, iterator=True, usecols=use_cols, dtype=dtypes), total=math.ceil(get_file_len(TMP_FILE)/CHUNK_SIZE)):

        # replace infinity values with NaN
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

        # drop all rows containing at least one NaN value
        chunk.dropna(inplace=True, axis=0)

        # pop label column -> no min-max scaling, put to the end of df
        labels = chunk.pop('label')

        # min-max scaling
        chunk = min_max_normalization(chunk, col_info)

        # encode categorical features
        chunk = one_hot_encoding(chunk, col_info)

        # append labels as last column
        chunk.reset_index(drop=True, inplace=True)
        chunk = chunk.assign(label=labels.reset_index(drop=True))

        all_chunks.append(chunk)

    df = pd.concat(all_chunks, ignore_index=True)
    STATS['n_dropped_samples'] = STATS['n_samples'] - df.shape[0]
    df = df.drop_duplicates(keep='first')
    STATS['n_duplicates'] = STATS['n_samples'] - STATS['n_dropped_samples'] - df.shape[0]
    STATS['n_dropped_samples'] += STATS['n_duplicates']

    df.to_csv(output_file, index=False)

    # remove tmp file
    os.remove(TMP_FILE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['UNSW-NB15', '5G-NIDD'], type=str)
    parser.add_argument(
        "-d", "--dataset_path", type=str, help="directory path of data set", required=True
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Path to store processed data file",
        type=str,
        required=True
    )

    args = parser.parse_args()

    with open("src/data/config.yml", "r", encoding='utf8') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.loader.SafeLoader)
    config = config[args.dataset]

    # set label name to 'label' if not specified
    config['label_name'] = 'label' if not config.get('label_name') else config.get('label_name')

    files = get_files(args.dataset_path)
    files.sort()

    col_info = prepare_col_info(files[0], config)

    col_info = first_iteration(files, col_info, config)

    output_file = str(args.dataset).lower().replace('-','').replace('_','') +'.csv'
    second_iteration(col_info, os.path.join(args.output_path, output_file))

    print(STATS)

if __name__ == "__main__":
    main()
