# -*- coding: utf-8 -*-
# create by Xu
# date 7/24/2020

import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



def getDataSet(fileName):
    # 读取数据
    data = pd.read_csv(fileName)
    dataSet = []
    for item in data.values:
        dataSet.append(list(item[:5]))
    label = list(data['K'])
    return dataSet, label


def divide(dataSet, labels):
    train_data, test_data, train_label, test_label = train_test_split(dataSet, labels, test_size=0.2)
    return train_data, test_data, train_label, test_label


# 随机森林
if __name__ == '__main__':
    data, label = getDataSet('./data/data_standard.csv')
    train_data, test_data, train_label, test_label = divide(data, label)
    clf = SVC(kernel='rbf')
    clf.fit(train_data, train_label)

    clf_pre_train = clf.predict(train_data)
    print('Training Set Evaluation F1-Score=>', f1_score(train_label, clf_pre_train, average='micro'))
    clf_pre_test = clf.predict(test_data)
    print('Testing Set Evaluation F1-Score=>', f1_score(test_label, clf_pre_test, average='micro'))

    cm = confusion_matrix(test_label, clf_pre_test)
    print(cm)
