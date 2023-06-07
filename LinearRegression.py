# 导入所需的库
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 加载示例数据集
boston = datasets.load_boston()
X = boston.data  # 特征
y = boston.target  # 目标

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
regression = LinearRegression()

# 在训练集上拟合模型
regression.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = regression.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("均方误差：", mse)
