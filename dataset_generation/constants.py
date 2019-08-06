class Constants:
    EPOCHS = 10
    BATCH_SIZE = 32
    NUM_BEHAVIORAL_FEATURES = 1
    BI_LSTM_OUT_DIM = 100
    SEED = 786
    VALIDATION_SPLIT = 0.1
    TRAIN_SIZE = 0.85
    MAX_FEATURES = 3000
    MAX_LEN = 300

    class Cols:
        ID = 'Review ID'
        REVIEW = 'Review'
        SENTIMENT = 'Sentiment Polarity'
        LABEL = 'Truth Value'

    PATH_TO_DATASET = '/Users/nikhilsulegaon/Downloads/BLT-C_Boulder_Lies_and_Truths_Corpus.csv'

    @staticmethod
    def label_value(label):
        label_to_num = {'T': 0, 'F': 1, 'D': 2}
        return label_to_num[label.upper()]