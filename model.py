from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, GRU
from keras.layers import concatenate, Input
import keras

max_len = 2048
dim = 300
option_num = 4

# argmi = Input()
argm = GRU(256, return_sequences=True, input_shape=(max_len, dim), name='argm')
argm = Dropout(0.1)(argm)
argm = TimeDistributed(Dense(256))(argm)
argm = GRU(256)(argm)
argm = Dense(128)(argm)
options = []

op1 = Input(shape=(max_len, dim), name='o1')
shared_lstm =LSTM(256)(op1)
op = Dense(128)(shared_lstm)
op = concatenate([argm, op])
op_out = Dense(64)(op)
options.append(op_out)

op2 = Input(shape=(max_len, dim), name='o2')
shared_lstm = LSTM(256)(op2)
op = Dense(128)(shared_lstm)
op = concatenate([argm, op])
op_out = Dense(64)(op)
options.append(op_out)

op3 = Input(shape=(max_len, dim), name='o3')
shared_lstm = LSTM(256)(op3)
op = Dense(128)(shared_lstm)
op = concatenate([argm, op])
op_out = Dense(64)(op)
options.append(op_out)

op4 = Input(shape=(max_len, dim), name='o4')
shared_lstm = LSTM(256)(op4)
op = Dense(128)(shared_lstm)
op = concatenate([argm, op])
op_out = Dense(64)(op)
options.append(op_out)


output = concatenate(options)
predictions = Dense(4, activation='sigmoid')(output)

model = Model(inputs=[argmi, op1, op2, op3, op4], outputs=predictions)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
"""
#input_a = Sequential()
#input_a.add(LSTM(128, return_sequences=False, input_shape=(max_len, dim)))
#input_a.add(TimeDistributed(Dense(256)))
#input_q = Sequential()
#input_q.add(LSTM(128, return_sequences=True, input_shape=(max_len, dim)))
#input_q.add(TimeDistributed(Dense(256)))


model = Sequential()
#model.add(Input(shape=(max_len, dim), dtype='int32'))
#model.add(keras.backend.concatenate([input_a, input_q], axis=0))
model.add(GRU(256, input_shape=(max_len, dim), return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(256)))
model.add(GRU(256))
model.add(Dense(128))
options = []
shared_lstm = LSTM(256, input_shape=(max_len, dim))
shared_lstm2 = LSTM(128)
for i in range(option_num):
    op = shared_lstm
    op = Dense(128)(op)
    op = concatenate([model, op])
    op = shared_lstm2(op)
    options.append(op)
model2 = Sequential()
model2.add(concatenate(options, axis=0))
model2.add(Dense(4, activation='sigmoid'))

model2.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
"""
