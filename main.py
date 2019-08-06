from keras import Input, Model
from keras.layers import Embedding, Bidirectional, TimeDistributed, Flatten, Dense, LSTM, Concatenate

from dataset_generation import gen_dataset
from dataset_generation.constants import Constants

treat_F_as_deceptive = False

dataset = gen_dataset(treat_F_as_deceptive)

output_dim_nn = 1 if treat_F_as_deceptive else 3
model_loss_function = 'binary_crossentropy' if treat_F_as_deceptive else 'categorical_crossentropy'

# Bidirectional LSTM to learn Linguistic features
bi_lstm_in = Input(shape=(Constants.MAX_LEN,))
lstm_model = Embedding(dataset.vocabulary_len(), Constants.BI_LSTM_OUT_DIM, input_length=Constants.MAX_LEN)(bi_lstm_in)
lstm_model = Bidirectional(LSTM(Constants.BI_LSTM_OUT_DIM, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(
    lstm_model)
lstm_model = TimeDistributed(Dense(Constants.BI_LSTM_OUT_DIM, activation='relu'))(lstm_model)
lstm_model = Flatten()(lstm_model)
bi_lstm_output = Dense(Constants.BI_LSTM_OUT_DIM, activation='relu')(lstm_model)

# 2-Layer NN to learn Behavioral features and output of Bi-LSTM to classify
nn_input_dim = Constants.NUM_BEHAVIORAL_FEATURES + Constants.BI_LSTM_OUT_DIM
nn_input = Input(shape=(Constants.NUM_BEHAVIORAL_FEATURES,))
nn_model = Concatenate()([nn_input, bi_lstm_output])
nn_model = Dense(12, input_dim=nn_input_dim, activation='relu')(nn_model)
nn_model = Dense(8, activation='relu')(nn_model)
nn_output = Dense(output_dim_nn, activation='sigmoid')(nn_model)

model = Model([bi_lstm_in, nn_input], nn_output)
model.compile(loss=model_loss_function, optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit([dataset.x_linguistic_train(), dataset.x_behavioral_train()], dataset.y_train(),
          batch_size=Constants.BATCH_SIZE,
          validation_split=Constants.VALIDATION_SPLIT,
          epochs=Constants.EPOCHS)

loss, accuracy = model.evaluate([dataset.x_linguistic_test(), dataset.x_behavioral_test()], dataset.y_test(),
                                batch_size=Constants.BATCH_SIZE,
                                verbose=2)

print('Test Accuracy: {:.3f}%'.format(accuracy * 100))
