# -*- coding:UTF-8 -*-
# create by Xu
# date 7/24/2020

from io import StringIO
import pandas as pd
import numpy as np
from dask_searchcv import GridSearchCV
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pydotplus


def cm_plot(y, yp):
    cm = confusion_matrix(y, yp)

    import matplotlib.pyplot as plt
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def getDataSet(fileName):
    # 读取数据
    data = pd.read_csv(fileName)
    data = data.drop(labels='ZMEMBER_NO', axis=1)
    dataSet = []
    for item in data.values:
        dataSet.append(list(item[:5]))
    label = list(data['K'])
    return dataSet, label


def divide(dataSet, labels):
    train_data, test_data, train_label, test_label = train_test_split(dataSet, labels, test_size=0.2)
    return train_data, test_data, train_label, test_label


# 决策树
if __name__ == '__main__':
    data, label = getDataSet('./data/data_standard.csv')
    train_data, test_data, train_label, test_label = divide(data, label)

    gini_thresholds = np.linspace(0, 0.5, 20)
    # parameters = {'splitter': ('best', 'random'), 'criterion': ("gini", "entropy"), "max_depth": [*range(4, 10)],
    #              'min_samples_leaf': [*range(1, 5, 50)], 'min_impurity_decrease': [*np.linspace(0, 0.5, 10)]}
    # 在不使用网格搜索时，基本很难界定

    clf = tree.DecisionTreeClassifier(max_depth=5, random_state=5)
    # GS = GridSearchCV(clf, parameters, cv=10)
    # clf = GS.fit(train_data, train_label)

    clf = clf.fit(train_data, train_label)

    scores = cross_val_score(clf, train_data, train_label)  # 存放模型的结果值
    print(scores)

    clf_pre_train = clf.predict(train_data)
    print('Training Set Evaluation F1-Score=>', f1_score(train_label, clf_pre_train, average='micro'))
    clf_pre_test = clf.predict(test_data)
    print('Testing Set Evaluation F1-Score=>', f1_score(test_label, clf_pre_test, average='micro'))

    cm = confusion_matrix(test_label, clf.predict(test_data))
    print(cm)

    # 可视化
    dataLabels = ['ZL', 'ZR', 'ZF', 'ZM', 'ZC', ]
    data_list = []
    data_dict = {}
    for each_label in dataLabels:
        for each in data:
            data_list.append(each[dataLabels.index(each_label)])
            data_dict[each_label] = data_list
        data_list = []
    lenses_pd = pd.DataFrame(data_dict)
    print(lenses_pd.keys())

    # 画决策树的决策流程
    dot_data = StringIO()
    print(clf.classes_)
    print(str(clf.classes_))
    tree.export_graphviz(clf, out_file=dot_data, feature_names=lenses_pd.keys(),
                         class_names=["0", "1"], filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("./data/tree.pdf")

    cm_plot(test_label, clf_pre_test).show()
