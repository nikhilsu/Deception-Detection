import nltk
import numpy as np
from keras.utils import to_categorical
from keras_preprocessing import text
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords, wordnet
from pandas import Series
from sklearn.model_selection import train_test_split

from dataset_generation.cleanup import ReviewsPreprocessor
from dataset_generation.constants import Constants
from dataset_generation.raw_data import RawData


class Dataset(object):
    def __init__(self, df_train, df_test, tokens, treat_F_as_deceptive):
        self.df_train = df_train
        self.df_test = df_test
        self.tokens = tokens
        self.treat_F_as_deceptive = treat_F_as_deceptive
        self.__behavioral_features = [Constants.Cols.SENTIMENT]

    def __get(self, column_list, train):
        data_frame = self.df_train if train else self.df_test
        data = data_frame[column_list]
        return np.asarray(data.tolist()) if isinstance(data, Series) else data.values

    def __categorize_if_needed(self, labels):
        if not self.treat_F_as_deceptive:
            return to_categorical(labels, 3)
        return labels

    def vocabulary_len(self):
        return len(self.tokens.word_index) + 1

    def x_linguistic_train(self):
        return self.__get(Constants.Cols.REVIEW, train=True)

    def x_linguistic_test(self):
        return self.__get(Constants.Cols.REVIEW, train=False)

    def x_behavioral_train(self):
        return self.__get(self.__behavioral_features, train=True)

    def x_behavioral_test(self):
        return self.__get(self.__behavioral_features, train=False)

    def y_train(self):
        return self.__categorize_if_needed(self.__get(Constants.Cols.LABEL, train=True))

    def y_test(self):
        return self.__categorize_if_needed(self.__get(Constants.Cols.LABEL, train=False))

    def num_of_behavioral_features(self):
        return len(self.__behavioral_features)


def gen_dataset(dataset_path, train_split, treat_F_as_deceptive):
    nltk.download('wordnet')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    generator = RawData()
    processor = ReviewsPreprocessor(text, stop_words, wordnet)
    vocab = Tokenizer(num_words=Constants.MAX_FEATURES)

    df = generator.generate(dataset_path, treat_F_as_deceptive)
    reviews = list(processor.process(df[Constants.Cols.REVIEW]))
    vocab.fit_on_texts(reviews)
    encoded_reviews = vocab.texts_to_sequences(reviews)
    df[Constants.Cols.REVIEW] = list(pad_sequences(encoded_reviews, maxlen=Constants.MAX_LEN))
    df_train, df_test = train_test_split(df, train_size=train_split,
                                         random_state=Constants.SEED)
    return Dataset(df_train, df_test, vocab, treat_F_as_deceptive)
