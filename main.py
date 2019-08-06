import argparse

from keras import Input, Model
from keras.layers import Embedding, Bidirectional, TimeDistributed, Flatten, Dense, LSTM, Concatenate

from dataset_generation import gen_dataset
from dataset_generation.constants import Constants


def train_and_evaluate(args):
    dataset = gen_dataset(args.path_to_dataset, args.train_split, args.treat_F_as_deceptive)

    bi_lstm_out_dim = args.output_dims
    num_behavioral_features = dataset.num_of_behavioral_features()
    output_dim_nn = 1 if args.treat_F_as_deceptive else 3
    model_loss_function = 'binary_crossentropy' if args.treat_F_as_deceptive else 'categorical_crossentropy'

    # Bidirectional LSTM to learn Linguistic features
    bi_lstm_in = Input(shape=(Constants.MAX_LEN,))
    lstm_model = Embedding(dataset.vocabulary_len(), bi_lstm_out_dim, input_length=Constants.MAX_LEN)(bi_lstm_in)
    lstm_model = Bidirectional(LSTM(bi_lstm_out_dim, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(
        lstm_model)
    lstm_model = TimeDistributed(Dense(bi_lstm_out_dim, activation='relu'))(lstm_model)
    lstm_model = Flatten()(lstm_model)
    bi_lstm_output = Dense(bi_lstm_out_dim, activation='relu')(lstm_model)

    # 2-Layer NN to learn Behavioral features and output of Bi-LSTM to classify
    nn_input_dim = num_behavioral_features + bi_lstm_out_dim
    nn_input = Input(shape=(num_behavioral_features,))
    nn_model = Concatenate()([nn_input, bi_lstm_output])
    nn_model = Dense(12, input_dim=nn_input_dim, activation='relu')(nn_model)
    nn_model = Dense(8, activation='relu')(nn_model)
    nn_output = Dense(output_dim_nn, activation='sigmoid')(nn_model)

    model = Model([bi_lstm_in, nn_input], nn_output)
    model.compile(loss=model_loss_function, optimizer='adam', metrics=['accuracy'])
    print('Model Summary:- \n\n')
    model.summary()

    model.fit([dataset.x_linguistic_train(), dataset.x_behavioral_train()], dataset.y_train(),
              batch_size=args.batch_size,
              validation_split=args.validation_split,
              epochs=args.epochs)

    loss, accuracy = model.evaluate([dataset.x_linguistic_test(), dataset.x_behavioral_test()], dataset.y_test(),
                                    batch_size=args.batch_size,
                                    verbose=2)

    print('Test Accuracy: {:.3f}%'.format(accuracy * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', required=True)
    parser.add_argument('--treat_F_as_deceptive', required=True)
    parser.add_argument('--batch_size', default=32, required=False)
    parser.add_argument('--epochs', default=1000, required=False)
    parser.add_argument('--train_split', default=0.85, required=False)
    parser.add_argument('--validation_split', default=0.05, required=False)
    parser.add_argument('--output_dims', default=100, required=False)

    cmd_args = parser.parse_args()

    train_and_evaluate(cmd_args)
