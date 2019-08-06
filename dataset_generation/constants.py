class Constants:
    SEED = 786
    MAX_FEATURES = 3000
    MAX_LEN = 300

    class Cols:
        ID = 'Review ID'
        REVIEW = 'Review'
        SENTIMENT = 'Sentiment Polarity'
        LABEL = 'Truth Value'

    @staticmethod
    def label_value(label):
        label_to_num = {'T': 0, 'F': 1, 'D': 2}
        return label_to_num[label.upper()]
