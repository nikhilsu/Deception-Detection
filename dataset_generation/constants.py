class Constants:
    SEED = 786
    MAX_FEATURES = 3000
    MAX_LEN = 300

    class Cols:
        REVIEW_COUNT = 'Review Count'
        WORKER_ID = 'Worker ID'
        REVIEW_LEN = 'Review Len'
        ID = 'Review ID'
        REVIEW = 'Review'
        SENTIMENT = 'Sentiment Polarity'
        LABEL = 'Truth Value'
        BEHAVIORAL_COLS = [SENTIMENT, REVIEW_COUNT, REVIEW_LEN]

    @staticmethod
    def label_value(label):
        label_to_num = {'T': 0, 'F': 1, 'D': 2}
        return label_to_num[label.upper()]
