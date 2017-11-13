# Sequentialモデルの最初のレイヤーとして
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))
# ここで model.output_shape == (None, 32)
# 注: `None`はバッチ次元．

# 2層目以降のレイヤーに対しては，入力サイズを指定する必要はありません:
model.add(LSTM(16))

# to stack recurrent layers, you must use return_sequences=True
# on any recurrent layer that feeds into another recurrent layer.
# note that you only need to specify the input size on the first layer.
model = Sequential()
model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(10))
