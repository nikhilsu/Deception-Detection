class Constants:
    max_features = 3000

    class Cols:
        ID = 'Review ID'
        REVIEW = 'Review'
        SENTIMENT = 'Sentiment Polarity'
        LABEL = 'Truth Value'

    PATH_TO_DATASET = '/Users/nikhilsulegaon/Downloads/BLT-C_Boulder_Lies_and_Truths_Corpus.csv'

    @staticmethod
    def label_value(label):
        label_to_num = {'T': 1, 'F': 2, 'D': 3}
        return label_to_num[label.upper()]