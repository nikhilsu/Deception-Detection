from keras import Input, Model
from keras.layers import Embedding, Bidirectional, TimeDistributed, Flatten, Dense, LSTM

from dataset_generation import gen_dataset
from dataset_generation.constants import Constants

dataset = gen_dataset()

# Bidirectional LSTM to learn Linguistic features
input_tensor = Input(shape=(Constants.MAX_LEN,))
model = Embedding(dataset.vocabulary_len(), 100, input_length=Constants.MAX_LEN)(input_tensor)
model = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(model)
model = TimeDistributed(Dense(100, activation='relu'))(model)
model = Flatten()(model)
output_tensor = Dense(100, activation='relu')(model)


model = Model(input_tensor, output_tensor)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
