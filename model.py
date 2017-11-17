from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, GRU
from keras.layers import concatenate, Input
import keras

max_len = 1024
oplen = 256
dim = 300
option_num = 4

argmi = Input(shape=(max_len, dim), name='argm')
argm = GRU(256, return_sequences=True)(argmi)
argm = Dropout(0.1)(argm)
argm = TimeDistributed(Dense(256))(argm)
argm = GRU(256)(argm)
argm = Dense(128)(argm)
options = []

shared_lstm = LSTM(256)
shered_op = Dense(128)
shered_out = Dense(64)

op1 = Input(shape=(oplen, dim), name='o1')
op = shared_lstm(op1)
op = shered_op(op)
op = concatenate([argm, op])
op_out = shered_out(op)
options.append(op_out)

op2 = Input(shape=(oplen, dim), name='o2')
op = shared_lstm(op2)
op = shered_op(op)
op = concatenate([argm, op])
op_out = shered_out(op)
options.append(op_out)

op3 = Input(shape=(oplen, dim), name='o3')
op = shared_lstm(op3)
op = shered_op(op)
op = concatenate([argm, op])
op_out = shered_out(op)
options.append(op_out)

op4 = Input(shape=(oplen, dim), name='o4')
op = shared_lstm(op4)
op = shered_op(op)
op = concatenate([argm, op])
op_out = shered_out(op)
options.append(op_out)


output = concatenate(options)
predictions = Dense(4, activation='softmax')(output)

model = Model(inputs=[argmi, op1, op2, op3, op4], outputs=predictions)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
