import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 读取垃圾邮件数据集
data = pd.read_csv('spam_dataset.csv')

# 提取邮件文本和标签
emails = data['email_text']
labels = data['label']

# 将邮件文本转换为向量表示
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 创建朴素贝叶斯分类器
nb_classifier = MultinomialNB()

# 在训练集上训练分类器
nb_classifier.fit(X_train, y_train)

# 在测试集上进行预测
accuracy = nb_classifier.score(X_test, y_test)

print("准确率:", accuracy)
