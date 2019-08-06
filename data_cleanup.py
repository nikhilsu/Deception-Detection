import nltk
from keras_preprocessing import text
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from constants import Constants
from data_generator import DatasetGenerator


class ReviewsPreprocessor(object):
    def __init__(self, tokenizer, stop_words, lemmatizer):
        self.tokenizer = tokenizer
        self.stopwords = stop_words
        self.lemmatizer = lemmatizer

    def __lemmatize(self, token):
        lemma = self.lemmatizer.morphy(token)
        return lemma if lemma else token

    def __tokenize(self, review):
        tokens = []
        for token in self.tokenizer.text_to_word_sequence(review):
            if token not in stopwords and len(token) > 1:
                tokens.append(self.__lemmatize(token))
        return tokens

    def process(self, reviews):
        return reviews.apply(self.__tokenize)


if __name__ == '__main__':
    nltk.download('wordnet')
    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))
    processor = ReviewsPreprocessor(text, stopwords, wordnet)
    generator = DatasetGenerator()
    df = generator.generate()
    df[Constants.Cols.REVIEW] = processor.process(df[Constants.Cols.REVIEW])
    print(df.head())
