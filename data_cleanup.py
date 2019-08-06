import numpy as np
import pandas as pd

from constants import Constants

# Constants
Cols = Constants.Cols


def remove_duplicates(data_frame):
    data_frame.drop_duplicates(subset=Cols.ID, keep=False, inplace=True)


def compute_and_add_metadata_to(data_frame):
    pass


def filter_rows_by_type_of_labels(data_frame):
    return data_frame[~data_frame[Cols.ID].str.startswith('[')]


def select_necessary_columns(data_frame):
    return data_frame[[Cols.REVIEW, Cols.SENTIMENT, Cols.LABEL]]


def transform_values_of_sentiment_columns(data_frame):
    data_frame[Cols.SENTIMENT] = np.where(dataset[Cols.SENTIMENT] == 'pos', 1, 0)


def __get_label_value(label):
    label_to_num = {'T': 1, 'F': 2, 'D': 3}
    return label_to_num[label.upper()]


def transform_values_of_label(data_frame, treat_F_as_deceptive):
    if treat_F_as_deceptive:
        data_frame[Cols.LABEL] = np.where(dataset[Cols.LABEL] == 'T', 1, 0)
    else:
        data_frame[Cols.LABEL] = data_frame[Cols.LABEL].map(__get_label_value)


if __name__ == '__main__':
    dataset = pd.read_csv(Constants.PATH_TO_DATASET)
    remove_duplicates(dataset)
    compute_and_add_metadata_to(dataset)
    dataset = filter_rows_by_type_of_labels(dataset)
    dataset = select_necessary_columns(dataset)
    transform_values_of_sentiment_columns(dataset)
    transform_values_of_label(dataset, treat_F_as_deceptive=False)
    print(dataset.head())
