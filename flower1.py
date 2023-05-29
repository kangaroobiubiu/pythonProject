import cv2
import numpy as np
from tensorflow import keras

# 加载训练好的鲜花分类模型
model = keras.models.load_model('flower_model.h5')

# 定义鲜花类别
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# 加载图像
image = cv2.imread('flower.jpg')

# 调整图像大小为模型所需尺寸
image = cv2.resize(image, (150, 150))
image = image / 255.0  # 归一化

# 添加一个维度以匹配模型输入形状
image = np.expand_dims(image, axis=0)

# 预测鲜花类别
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

# 获取预测结果标签
flower_class = classes[predicted_class]

# 在图像上显示预测结果
cv2.putText(image, flower_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 显示结果图像
cv2.imshow('Flower Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
