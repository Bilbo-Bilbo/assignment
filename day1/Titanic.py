# -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# load data
import pandas as pd
data = pd.read_csv('train.csv')
df = data.copy()
print('数据基本信息：')
df.info()
print('\n数据样例：')
df.sample(10)

# %%
# delete some features that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
print('\n删除无用特征后的基本信息：')
df.info()

# %%
# check if there is any NaN in the dataset
print('\n数据缺失值检查：')
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
df.dropna(inplace=True)
print('Is there any NaN in the dataset after dropping: {}'.format(df.isnull().values.any()))

# %%
# convert categorical data into numerical data using one-hot encoding
# For example, a feature like sex with categories ['male', 'female'] would be transformed into two new binary features, sex_male and sex_female, represented by 0 and 1.
df = pd.get_dummies(df)
print('\n独热编码后的样例数据：')
df.sample(10)

# %% 
# separate the features and labels
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# %%
# train-test split
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1919810)
print('\n数据集分割结果：')
print('X_train: {}'.format(np.shape(X_train)))
print('y_train: {}'.format(np.shape(y_train)))
print('X_test: {}'.format(np.shape(X_test)))
print('y_test: {}'.format(np.shape(y_test)))

# %%
# build model
# build three classification models
# SVM, KNN, Random Forest
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.svm import SVC # type: ignore

models = dict()
models['SVM'] = SVC(kernel='linear')   # kernel specifies the kernel type, 'linear' is used here
models['KNeighbor'] = KNeighborsClassifier(n_neighbors=25, weights='distance')  # n_neighbors specifies the number of neighbors
models['RandomForest'] = RandomForestClassifier(n_estimators=300, max_depth=9, max_features=0.60, random_state=114514)  # increased n_estimators to 300

# %%
# define functions to evaluate the models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore
from matplotlib import pyplot as plt
import numpy as np

def plot_cm(model, y_true, y_pred, name=None):
    """plot the confusion matrix
    :param model: the classification model
    :param y_true: the true labels
    :param y_pred: the predicted labels
    :param name: the name of the model
    """
    _, ax = plt.subplots()
    if name is not None:
        ax.set_title(name)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.show()
    return None


def plot_cm_ratio(model, y_true, y_pred, name=None):
    """plot the confusion matrix (normalized)
    :param model: the classification model
    :param y_true: the true labels
    :param y_pred: the predicted labels
    :param name: the name of the model
    """
    _, ax = plt.subplots()
    if name is not None:
        ax.set_title(name)
    cm = confusion_matrix(y_true, y_pred)
    cm_ratio = np.zeros(cm.shape)
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            cm_ratio[i, j] = cm[i, j] / cm[i].sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ratio, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.show()
    return None


def model_perf(model, y_true, y_pred, name=None):
    """Print model accuracy, TPR, and FPR
    """
    if name is not None:
        print('For model {}: \n'.format(name))
    cm = confusion_matrix(y_true, y_pred)
    for i in range(len(model.classes_)):
        # tp: true positive, fp: false positive, fn: false negative, tn: true negative
        # tpr: true positive rate, fpr: false positive rate, acc: accuracy
        tp = cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tn = cm.sum() - tp - fp - fn
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        acc = (tp + tn) / cm.sum()
        print('For class {}: \n TPR is {:.4f}; \n FPR is {:.4f}; \n ACC is {:.4f}. \n'
              .format(model.classes_[i], tpr, fpr, acc))
    return None


def ovo_eval(model, name=None):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    plot_cm(model, y_test, prediction, name)
    plot_cm_ratio(model, y_test, prediction, name)
    model_perf(model, y_test, prediction, name)
    score = model.score(X_test, y_test)
    print('Overall Accuracy: {:.4f}'.format(score))
    return score

# %%
# predict and evaluate
print('\n模型评估结果：')
record = dict()
for name, model in models.items():
    print(f'\n=== {name} 模型 ===')
    record[name] = ovo_eval(model, name)

# 比较模型性能
print('\n=== 模型性能比较 ===')
for name, score in sorted(record.items(), key=lambda x: x[1], reverse=True):
    print(f'{name}: {score:.4f}')