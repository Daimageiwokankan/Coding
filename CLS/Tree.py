# -*- coding: utf-8 -*-


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
from IPython.display import Image
from sklearn import tree

from CLS.FeatureSelection import FeatureSelection


class Tree(object):  # 1.初始化建立决策树
    def __init__(self, DataFrame, max_depth=3):
        # 特征选择
        new_order = np.random.permutation(len(DataFrame))
        new_df = DataFrame.take(new_order)
        f = FeatureSelection(new_df)
        self.column_names = f.return_feature_set(variance_threshold=True, tree_select=True, rlr_select=True)
        feature = new_df[self.column_names]
        target = new_df['流失标志']
        # 准备训练和测试数据
        breakPoint = math.floor(len(feature) * 0.2)
        feature_Train = feature[:-breakPoint]
        target_Train = target[:-breakPoint]
        self.feature_Test = feature[-breakPoint:]
        self.target_Test = target[-breakPoint:]
        # 分类决策树
        self.clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=6,
                                               min_samples_leaf=3, class_weight='balanced', presort=True)
        self.clf = self.clf.fit(feature_Train, target_Train)
        print('模型准确率为：%2.2f %%' % (self.clf.score(feature_Train, target_Train) * 100))
        print('决策完成！')

    def Tree_assess(self):  # 2.模型准确率评价
        pre_Test = self.clf.predict(self.feature_Test)
        df_result = pd.DataFrame(np.column_stack((pre_Test, np.array(self.target_Test))),
                                 columns=['$C-流失标志', '流失标志'])

        def a(x):
            if x[0] == x[1]:
                return 1
            else:
                return 0

        df_result['判断'] = df_result.apply(a, axis=1)
        preRate = df_result['判断'].groupby(df_result['流失标志']).mean()
        preRT = df_result['判断'].mean()
        preCount = df_result.groupby([df_result['流失标志'], df_result['判断']]).count()
        DT = pd.DataFrame([[preCount['$C-流失标志'][0.0, 0], preCount['$C-流失标志'][0.0, 1], preRate[0.0]],
                           [preCount['$C-流失标志'][1.0, 0], preCount['$C-流失标志'][1.0, 1], preRate[1.0]],
                           [preCount['$C-流失标志'][0.0, 0] + preCount['$C-流失标志'][1.0, 0],
                            preCount['$C-流失标志'][0.0, 1] + preCount['$C-流失标志'][1.0, 1], preRT]],
                          columns=['错误数', '正确数', '准确率'], index=[0, 1, '总计']).round(2)
        return DT

    def Tree_view(self):  # 3.1模型结果透视化
        transform = pd.read_csv('中英互释.csv', engine='python', index_col='中文名')
        dot_data = tree.export_graphviz(self.clf, out_file=None,
                                        feature_names=list(transform['英文名'][self.feature_Test.columns]),
                                        class_names=['0', '1'],
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("TreeView.pdf")
        return Image(graph.create_png())

    def Tree_feat_import(self):  # 3.2特征变量重要性
        clf = self.clf
        feature_name = self.column_names
        feature_var = list(clf.feature_importances_)  # feature scores #
        features = dict(zip(feature_name, feature_var))
        features = dict(sorted(features.items(), key=lambda d: d[1]))
        index = list(features.keys())
        values = list(features.values())
        plt.figure(figsize=(13, 7))
        plt.title('决策树特征重要性条形图', fontsize=24)
        plt.barh(index, values, alpha=0.7)
        plt.xlabel('特征重要性', color='gray', fontsize=16)
        plt.ylabel('特征变量名', color='gray', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gcf().savefig('Tree_feature_importances.png')
        plt.show()


def getDataSet(fileName):
    # 读取数据
    data = pd.read_csv(fileName)
    data = data.drop(labels='ZMEMBER_NO', axis=1)
    dataSet = []
    for item in data.values:
        dataSet.append(list(item[:5]))
    lable = list(data['K'])
    return dataSet, lable
