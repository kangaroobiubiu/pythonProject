import numpy as np
from sklearn.linear_model import LinearRegression

# 输入数据，每行表示一个样本，每列表示一个特征
# 例如，第一行表示第一个房子的面积和卧室数量，第二行表示第二个房子的面积和卧室数量，依此类推
# 这里使用了面积和卧室数量作为特征，你可以根据实际情况调整特征
X = np.array([[100, 2],
              [150, 3],
              [120, 2],
              [170, 4],
              [200, 3]])

# 对应的房价，每行表示一个样本的房价
# 这里使用了随机生成的房价，你可以根据实际情况替换成真实数据
y = np.array([250, 400, 300, 500, 550])

# 创建线性回归模型
model = LinearRegression()

# 使用输入数据和对应的房价训练模型
model.fit(X, y)

# 预测新样本的房价
new_sample = np.array([[130, 2]])  # 假设有一个新样本，面积为130平方米，卧室数量为2
predicted_price = model.predict(new_sample)

print("预测的房价:", predicted_price)
