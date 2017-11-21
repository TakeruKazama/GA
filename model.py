from keras.models import Model
from keras.layers import Dense, BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers import GRU
from keras.layers import add, Input


max_len = 2048
dim = 300
option_num = 4
layer_num = 10
grus = []

shered_gru = Bidirectional(GRU(64))

argmi = Input(shape=(max_len, dim))
argm = Bidirectional(GRU(128, return_sequences=True))(argmi)
argm = BatchNormalization()(argm)
grus.append(shered_gru(argm))
for i in range(layer_num):
    argm = Bidirectional(GRU(128, return_sequences=True))(argm)
    argm = BatchNormalization()(argm)
    grus.append(shered_gru(argm))

output = add(grus)
predictions = Dense(option_num, activation='softmax')(output)

model = Model(inputs=argmi, outputs=predictions)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
