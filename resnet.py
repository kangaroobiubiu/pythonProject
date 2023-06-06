import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# 加载预训练的ResNet50模型
model = ResNet50(weights='imagenet')

# 加载图像并进行预处理
img_path = 'cat.jpg'  # 图像文件路径
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 使用ResNet50模型进行图像分类
predictions = model.predict(x)
decoded_predictions = decode_predictions(predictions, top=3)[0]

print('预测结果：')
for _, label, probability in decoded_predictions:
    print(f'{label}: {probability * 100}%')
