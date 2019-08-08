class Constants:
    SEED = 786
    MAX_FEATURES = 3000
    MAX_LEN = 300

    class Cols:
        MEAN_SIM_SCORE = 'Mean Sim Score'
        STD_SIM_SCORE = 'Std Sim Score'
        TIME_TO_WRITE_REVIEW = 'Time to Write a Review Pair (sec.)'
        DOMAIN = 'Domain'
        REVIEW_COUNT = 'Review Count'
        WORKER_ID = 'Worker ID'
        REVIEW_LEN = 'Review Len'
        ID = 'Review ID'
        REVIEW = 'Review'
        SENTIMENT = 'Sentiment Polarity'
        LABEL = 'Truth Value'
        BEHAVIORAL_COLS = [SENTIMENT, DOMAIN, REVIEW_COUNT, REVIEW_LEN, MEAN_SIM_SCORE, STD_SIM_SCORE, TIME_TO_WRITE_REVIEW]

    @staticmethod
    def label_value(label):
        label_to_num = {'T': 0, 'F': 1, 'D': 2}
        return label_to_num[label.upper()]
