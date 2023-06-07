# 导入所需的库
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing import sequence
from keras.datasets import imdb
# Here's an example of using the Keras library with TensorFlow backend to build and train a recurrent neural network (RNN) for sentiment analysis:

# 定义超参数
max_features = 5000
max_len = 100
batch_size = 32
embedding_dims = 50
epochs = 5

# 加载IMDB电影评论数据集
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# 对序列进行填充/截断，使其具有相同的长度
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# 创建RNN模型
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print("损失：", loss)
print("准确率：", accuracy)
