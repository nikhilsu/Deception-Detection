import numpy as np
import pandas as pd

# Constants

ID = 'Review ID'
REVIEW = 'Review'
SENTIMENT = 'Sentiment Polarity'
LABEL = 'Truth Value'
PATH_TO_DATASET = '/Users/nikhilsulegaon/Downloads/BLT-C_Boulder_Lies_and_Truths_Corpus.csv'


def select_necessary_columns(data_frame):
    return data_frame[[REVIEW, SENTIMENT, LABEL]]


def compute_and_add_metadata_to(data_frame):
    return data_frame


def remove_duplicates(data_frame):
    data_frame.drop_duplicates(subset=ID, keep=False, inplace=True)


def filter_rows_by_type_of_labels(data_frame):
    return data_frame[~data_frame['Review ID'].str.startswith('[')]


def transform_values_of_sentiment_columns(data_frame):
    data_frame[SENTIMENT] = np.where(dataset[SENTIMENT] == 'pos', 1, 0)


def get_label_value(label):
    label_to_num = {'T': 1, 'F': 2, 'D': 3}
    return label_to_num[label.upper()]


def transform_values_of_label(data_frame, treat_F_as_deceptive=True):
    if treat_F_as_deceptive:
        data_frame[LABEL] = np.where(dataset[LABEL] == 'T', 1, 0)
    else:
        data_frame[LABEL] = data_frame[LABEL].map(get_label_value)


dataset = pd.read_csv(PATH_TO_DATASET)
