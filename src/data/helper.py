import typing
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def get_files(data_set_dir: str) -> typing.List[str]:
    """Reads given directory and returns each file with path from the directory

    Parameters
    ----------
    data_set_dir : str
        Directory path to read in

    Returns
    -------
    typing.List[str]
        List of paths of the files in the directory
    """
    filenames = []
    for path in Path(data_set_dir).rglob('*.csv'):
        filenames.append(path.resolve())
    if len(filenames) < 1:
        # nsl kdd has .txt files
        for path in Path(data_set_dir).rglob('*.txt'):
            filenames.append(path.resolve())

    return filenames

def get_file_len(fname: str) -> int:
    """Reads and returns number of lines in the file

    Parameters
    ----------
    fname : str
        path to file

    Returns
    -------
    int
        Number of lines in file

    Raises
    ------
    IOError
    """
    # from https://www.kaggle.com/code/szelee/how-to-import-a-csv-file-of-55-million-rows
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])+1

def one_hot_encoding(dataset: pd.DataFrame, col_info) -> pd.DataFrame:
    """One hot encoding of every categorical column in dataframe
    Parameters
    ----------
    dataset : pd.DataFrame
        data set
    Returns
    -------
    pd.DataFrame
        dataframe without categorical columns
    """

    for categorical_column in list(
        set(dataset.columns) - set(dataset.select_dtypes(include=np.number).columns)
    ):
        # get unique values to create column names
        columns = [
            categorical_column + "_" + x for x in col_info[categorical_column]['uniques']
        ]

        # define one hot encoding
        encoder = OneHotEncoder(sparse_output=False, categories=[col_info[categorical_column]['uniques']])
        # transform data
        encoder_df = pd.DataFrame(
            encoder.fit_transform(
                np.asarray(dataset[categorical_column]).reshape(-1, 1)
            )
        )
        # set column names for one hot encoded data set
        encoder_df.columns = columns

        # merge one hot encoded data into data set
        dataset = pd.concat([dataset.reset_index(drop=True), encoder_df], axis=1)

        # drop categorical column
        dataset.drop(categorical_column, axis=1, inplace=True)

    # return sorted dataframe
    return dataset.reindex(sorted(dataset.columns), axis=1)

def get_min_dtype(x: np.number, y: np.number) -> np.dtype:
    """Returns the minimal fitting dtype x and y fits in

    Parameters
    ----------
    x : np.number
    y : np.number

    Returns
    -------
    np.dtype
        minimal dtype both x and y fit into
    """

    min_x_dtype = np.min_scalar_type(x)
    min_y_dtype = np.min_scalar_type(y)
    min_type = np.promote_types(min_x_dtype, min_y_dtype)
    return min_type

def min_max_normalization(dataset: pd.DataFrame, col_info: typing.Dict[str, dict]) -> pd.DataFrame:
    """Min-Max normalization of dataset
    Parameters
    ----------
    dataset : pd.DataFrame
        data set not normalized
    Returns
    -------
    pd.DataFrame
        normalized dataset in range between 0 and 1
    """

    for numeric_column in list(
        dataset.select_dtypes(include=np.number).columns
    ):
        col_min, col_max = col_info[numeric_column]['min'], col_info[numeric_column]['max']
        dataset[numeric_column] = (dataset[numeric_column] - col_min) / (col_max - col_min)

    return dataset
