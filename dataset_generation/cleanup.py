class ReviewsPreprocessor(object):
    def __init__(self, tokenizer, stopwords, lemmatizer):
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.lemmatizer = lemmatizer

    def __lemmatize(self, token):
        lemma = self.lemmatizer.morphy(token)
        return lemma if lemma else token

    def __clean(self, review):
        tokens = []
        for token in self.tokenizer.text_to_word_sequence(review):
            if token not in self.stopwords and len(token) > 1:
                tokens.append(self.__lemmatize(token))
        return ' '.join(tokens)

    def process(self, reviews):
        return reviews.apply(self.__clean)
