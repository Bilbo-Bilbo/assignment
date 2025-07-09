import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from random import randint
import joblib

# 读取CSV数据，列名为'PassengerId'的列作为行名称
df = pd.read_csv('train.csv', index_col='PassengerId')

# 删除缺失数据的行
df = df.dropna()

print(df.shape)


# 删除无用特征
df = df.drop(columns=['Name', 'Ticket', 'Cabin'], axis=1)

# 创建LabelEncoder对象
le = LabelEncoder()

# 对包含字符串数据的列进行编码
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# 分离目标值与特征
X = df.drop('Survived', axis=1)
y = df['Survived']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randint(0,100))

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 逻辑回归
model = LogisticRegression()
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, '泰坦尼克.joblib')

# 打印模型信息
print('Model coefficients:', model.coef_)
print('Model intercept:', model.intercept_)

# 预测
y_pred = model.predict(X_test)

print(y_pred)
print(y_test)

# 模型评估
print(classification_report(y_test, y_pred))
