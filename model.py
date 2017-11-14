from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM, GRU
from keras.layers import concatenate, Input

max_len = 2048
dim = 300
layer_num = 2
option_num = 4

input_a = Input(shape=(max_len, dim))
input_a = LSTM(128, return_sequences=True)(input_a)

input_q = Input(shape=(max_len, dim))
# input_q = Embedding(max_len, 256)(input_q)
input_q = LSTM(128, return_sequences=True)(input_q)


model = Sequential()
model.add(concatenate([input_a, input_q], axis=0))
for i in range(layer_num):
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.2))
options = []
for i in range(option_num):
    op = Input(shape=(max_len, dim))
    op = Embedding(256)(op)
    op = concatenate([model, op], axis=0)
    op = LSTM(64, return_sequences=True)(op)
    options.append(op)
model2 = Sequential()
model2.add(concatenate(options, axis=0))
model2.add(Dense(4, activation='sigmoid'))

model2.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

"""

"""
