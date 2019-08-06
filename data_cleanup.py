import nltk
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

from constants import Constants

stopwords = set(stopwords.words('english'))
nltk.download('wordnet')
nltk.download('stopwords')


def tokenize(review):
    tokenizer = Tokenizer(num_words=Constants.max_features)
    tokens = []
    for token in tokenizer.texts_to_sequences(review):
        if token not in stopwords and len(token) > 1:
            tokens.append(lemmatize(token))
    return tokens


def lemmatize(token):
    lemma = wn.morphy(token)
    return lemma if lemma else token
