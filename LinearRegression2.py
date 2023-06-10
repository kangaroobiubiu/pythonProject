# 导入所需的库
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 创建一个示例数据集
X = np.array([[1, 2, 3, 4, 5]]).T  # 特征矩阵，这里只有一个特征
y = np.array([2, 4, 6, 8, 10])  # 目标变量

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算均方误差（Mean Squared Error）
mse = mean_squared_error(y_test, y_pred)
print("均方误差:", mse)
