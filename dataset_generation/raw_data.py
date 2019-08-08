import gensim.downloader as api
import numpy as np
import pandas as pd

from dataset_generation.constants import Constants

# Constants
Cols = Constants.Cols


class RawData(object):
    def __init__(self):
        self.word_vectors = api.load('glove-wiki-gigaword-100')

    @staticmethod
    def __remove_duplicates(data_frame):
        data_frame.drop_duplicates(subset=Cols.ID, keep=False, inplace=True)

    @staticmethod
    def __filter_rows_by_type_of_labels(data_frame):
        return data_frame[~data_frame[Cols.ID].str.startswith('[')]

    # TODO: Needs to be implemented to add Behavioral Features
    @staticmethod
    def __select_necessary_columns(data_frame):
        return data_frame[[Cols.REVIEW] + Cols.BEHAVIORAL_COLS + [Cols.LABEL]]

    @staticmethod
    def __transform_values_of_sentiment_columns(data_frame):
        data_frame[Cols.SENTIMENT] = np.where(data_frame[Cols.SENTIMENT] == 'pos', 1, 0)

    @staticmethod
    def __transform_values_of_domain_columns(data_frame):
        data_frame[Cols.DOMAIN] = np.where(data_frame[Cols.DOMAIN] == 'Hotels', 1, 0)

    @staticmethod
    def __transform_values_of_label(data_frame, treat_F_as_deceptive):
        if treat_F_as_deceptive:
            data_frame[Cols.LABEL] = np.where(data_frame[Cols.LABEL] == 'T', 1, 0)
        else:
            data_frame[Cols.LABEL] = data_frame[Cols.LABEL].map(Constants.label_value)

    @staticmethod
    def __normalize_series(data_frame, label):
        series = data_frame[label]
        data_frame[label] = (series - series.mean()) / series.std()

    def __find_doc_sim_score(self, reviews):
        sim_scores = []
        for i in range(len(reviews) - 1):
            for j in range(i + 1, len(reviews)):
                sim_scores.append(self.word_vectors.wmdistance(reviews[i], reviews[j]))
        return np.mean(sim_scores), np.std(sim_scores)

    def __compute_and_add_metadata_to(self, data_frame):
        # Adding Review Length
        data_frame[Cols.REVIEW_LEN] = data_frame[Cols.REVIEW].str.len()

        # Adding Review count per worker
        group_by = data_frame.groupby(Cols.WORKER_ID)[Cols.REVIEW].count()
        data_frame[Cols.REVIEW_COUNT] = data_frame.join(group_by, on=Cols.WORKER_ID, rsuffix='_r')[(Cols.REVIEW + '_r')]

        # Adding Review Similarity Scores per User and Domain
        sim_score_grouping = data_frame.groupby([Cols.WORKER_ID, Cols.DOMAIN])[Cols.REVIEW].apply(
            lambda x: self.__find_doc_sim_score(x.values))

        sim_tuples = data_frame.join(sim_score_grouping, on=[Cols.WORKER_ID, Cols.DOMAIN], rsuffix='_r')

        data_frame[[Cols.MEAN_SIM_SCORE, Cols.STD_SIM_SCORE]] = pd.DataFrame(sim_tuples[Cols.REVIEW + '_r'].tolist(),
                                                                             index=data_frame.index)
        self.__normalize_series(data_frame, Cols.MEAN_SIM_SCORE)
        self.__normalize_series(data_frame, Cols.STD_SIM_SCORE)


    def generate(self, path, treat_F_as_deceptive):
        dataset = pd.read_csv(path)
        self.__remove_duplicates(dataset)
        self.__compute_and_add_metadata_to(dataset)
        dataset = self.__filter_rows_by_type_of_labels(dataset)
        dataset = self.__select_necessary_columns(dataset)
        self.__transform_values_of_sentiment_columns(dataset)
        self.__transform_values_of_domain_columns(dataset)
        self.__transform_values_of_label(dataset, treat_F_as_deceptive=treat_F_as_deceptive)
        dataset.reset_index(drop=True)
        return dataset
