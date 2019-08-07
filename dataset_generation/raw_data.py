import numpy as np
import pandas as pd

from dataset_generation.constants import Constants

# Constants
Cols = Constants.Cols


class RawData(object):

    @staticmethod
    def __remove_duplicates(data_frame):
        data_frame.drop_duplicates(subset=Cols.ID, keep=False, inplace=True)

    # TODO: Needs to be implemented to add Behavioral Features
    @staticmethod
    def __compute_and_add_metadata_to(data_frame):
        # Adding Review Length
        data_frame[Cols.REVIEW_LEN] = data_frame[Cols.REVIEW].str.len()
        # Adding Review count per worker
        group_by = data_frame.groupby(Cols.WORKER_ID)[Cols.REVIEW].count()
        data_frame[Cols.REVIEW_COUNT] = data_frame.join(group_by, on=Cols.WORKER_ID, rsuffix='_r')[(Cols.REVIEW + '_r')]

    @staticmethod
    def __filter_rows_by_type_of_labels(data_frame):
        return data_frame[~data_frame[Cols.ID].str.startswith('[')]

    @staticmethod
    def __select_necessary_columns(data_frame):
        return data_frame[[Cols.REVIEW] + Cols.BEHAVIORAL_COLS + [Cols.LABEL]]

    @staticmethod
    def __transform_values_of_sentiment_columns(data_frame):
        data_frame[Cols.SENTIMENT] = np.where(data_frame[Cols.SENTIMENT] == 'pos', 1, 0)

    @staticmethod
    def __transform_values_of_label(data_frame, treat_F_as_deceptive):
        if treat_F_as_deceptive:
            data_frame[Cols.LABEL] = np.where(data_frame[Cols.LABEL] == 'T', 1, 0)
        else:
            data_frame[Cols.LABEL] = data_frame[Cols.LABEL].map(Constants.label_value)

    def generate(self, path, treat_F_as_deceptive):
        dataset = pd.read_csv(path)
        self.__remove_duplicates(dataset)
        self.__compute_and_add_metadata_to(dataset)
        dataset = self.__filter_rows_by_type_of_labels(dataset)
        dataset = self.__select_necessary_columns(dataset)
        self.__transform_values_of_sentiment_columns(dataset)
        self.__transform_values_of_label(dataset, treat_F_as_deceptive=treat_F_as_deceptive)
        return dataset
