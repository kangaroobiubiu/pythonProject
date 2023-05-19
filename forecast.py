import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 读取气温数据
data = pd.read_csv('temperature_data.csv')

# 将日期列设置为索引
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 拟合ARIMA模型
model = ARIMA(data, order=(1, 1, 1))  # 使用(1, 1, 1)作为ARIMA模型的参数
model_fit = model.fit()

# 进行气温预测
forecast = model_fit.predict(start=len(data), end=len(data)+6)  # 预测未来7天的气温

# 打印预测结果
print(forecast)
