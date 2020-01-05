import os
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

# 数据加载
train=pd.read_csv('/Users/mike_liu/Desktop/数据分析45讲/titan生存预测/train.csv', engine='python',encoding='utf-8')
test_data = pd.read_csv('/Users/mike_liu/Desktop/数据分析45讲/titan生存预测/test.csv', engine='python',encoding='utf-8')
# 数据探索
print(train.info())
print('-'*30)
print(train.describe())
print('-'*30)
print(train.describe(include=['O']))
print('-'*30)
print(train.head())
print('-'*30)
print(train.tail())
# 数据清洗
# 使用平均年龄来填充年龄中的 nan 值
train['Age'].fillna(train['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# 使用票价的均值填充票价中的 nan 值
train['Fare'].fillna(train['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
print(train['Embarked'].value_counts())

# 使用登录最多的港口来填充登录港口的 nan 值
train['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train[features]
train_labels = train['Survived']
test_features = test_data[features]

dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features, train_labels)

test_features=dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
pred_labels = clf.predict(test_features)

# 得到决策树准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score 准确率为 %.4lf' % acc_decision_tree)