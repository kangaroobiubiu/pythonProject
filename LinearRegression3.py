import numpy as np
from sklearn.linear_model import LinearRegression

# 输入特征
X = np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350], [2450], [1425], [1700]])
# 输出标签
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X, y)

# 预测房价
X_test = np.array([[2000]])
predicted_price = model.predict(X_test)

# 输出预测结果
print("预测的房价：$%.2f" % predicted_price)
