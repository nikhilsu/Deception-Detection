import nltk
from keras_preprocessing import text
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords, wordnet
from sklearn.model_selection import train_test_split

from dataset_generation.cleanup import ReviewsPreprocessor
from dataset_generation.constants import Constants
from dataset_generation.raw_data import RawData


def main():
    nltk.download('wordnet')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    generator = RawData()
    processor = ReviewsPreprocessor(text, stop_words, wordnet)
    t = Tokenizer(num_words=Constants.MAX_FEATURES)
    df = generator.generate()
    reviews = list(processor.process(df[Constants.Cols.REVIEW]))
    t.fit_on_texts(reviews)
    encoded_reviews = t.texts_to_sequences(reviews)
    df[Constants.Cols.REVIEW] = list(pad_sequences(encoded_reviews))
    df_train, df_test = train_test_split(df, train_size=Constants.TRAIN_SIZE, random_state=Constants.SEED)
    print(df.head())


main()
