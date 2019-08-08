import argparse


def train_and_evaluate(args):
    bi_lstm_out_dim = args.output_dims
    num_behavioral_features = len(Constants.Cols.BEHAVIORAL_COLS)
    output_dim_nn = 1 if args.treat_F_as_deceptive else 3
    model_loss_function = 'binary_crossentropy' if args.treat_F_as_deceptive else 'categorical_crossentropy'
    cross_validation_accuracies = []
    for i, dataset in enumerate(gen_dataset(args.path_to_dataset, args.k_folds, args.treat_F_as_deceptive)):
        # Bidirectional LSTM to learn Linguistic features
        bi_lstm_in = Input(shape=(Constants.MAX_LEN,))
        lstm_model = Embedding(dataset.vocabulary_len(), bi_lstm_out_dim, input_length=Constants.MAX_LEN)(bi_lstm_in)
        lstm_model = Bidirectional(LSTM(bi_lstm_out_dim, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(
            lstm_model)
        lstm_model = TimeDistributed(Dense(bi_lstm_out_dim, activation='relu'))(lstm_model)
        lstm_model = MaxPool1D(3)(lstm_model)
        lstm_model = Flatten()(lstm_model)
        bi_lstm_output = Dense(bi_lstm_out_dim, activation='relu')(lstm_model)

        if args.use_bi_lstm_only:
            main_model_input = bi_lstm_in
            main_model_output = Dense(output_dim_nn, activation='sigmoid')(bi_lstm_output)
            dataset_input = dataset.x_linguistic_train()

        else:
            # 2-Layer NN to learn Behavioral features and output of Bi-LSTM to classify
            nn_input_dim = num_behavioral_features + bi_lstm_out_dim
            nn_input = Input(shape=(num_behavioral_features,))
            nn_model = Concatenate()([nn_input, bi_lstm_output])
            nn_model = Dense(12, input_dim=nn_input_dim, activation='relu')(nn_model)
            nn_model = Dropout(0.25)(nn_model)
            nn_model = Dense(8, activation='relu')(nn_model)
            nn_output = Dense(output_dim_nn, activation='sigmoid')(nn_model)
            main_model_input = [bi_lstm_in, nn_input]
            main_model_output = nn_output
            dataset_input = [dataset.x_linguistic_train(), dataset.x_behavioral_train()]

        model = Model(main_model_input, main_model_output)
        model.compile(loss=model_loss_function, optimizer='adam', metrics=['accuracy'])
        print('Model Summary:- \n\n')
        model.summary()

        # For model visualization:-
        # from keras.utils import plot_model
        # plot_model(model, to_file='model.png')

        model.fit(dataset_input, dataset.y_train(),
                  batch_size=args.batch_size,
                  validation_split=args.validation_split,
                  epochs=args.epochs)

        loss, accuracy = model.evaluate([dataset.x_linguistic_test(), dataset.x_behavioral_test()], dataset.y_test(),
                                        batch_size=args.batch_size,
                                        verbose=2)

        print('Test Accuracy CV - {}: {:.3f}%'.format(i, accuracy * 100))
        cross_validation_accuracies.append(accuracy * 100)

    print("Avg Test Acc after Cross Validation: {:.2f}% +/-{:.2f}%".format(np.mean(cross_validation_accuracies),
                                                                           np.std(cross_validation_accuracies)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', required=True)
    parser.add_argument('--treat_F_as_deceptive', required=True)
    parser.add_argument('--use_bi_lstm_only', default=False, required=False,
                        help='Use this option to train only the Bi-directional LSTM model with the dataset')
    parser.add_argument('--batch_size', default=32, required=False)
    parser.add_argument('--epochs', default=50, required=False)
    parser.add_argument('--k_folds', default=10, required=False)
    parser.add_argument('--validation_split', default=0.1, required=False)
    parser.add_argument('--output_dims', default=100, required=False)

    cmd_args = parser.parse_args()

    import numpy as np
    from keras import Input, Model
    from keras.layers import Embedding, Bidirectional, TimeDistributed
    from keras.layers import Flatten, Dense, LSTM, Concatenate, Dropout, MaxPool1D
    from dataset_generation import gen_dataset
    from dataset_generation.constants import Constants

    train_and_evaluate(cmd_args)
