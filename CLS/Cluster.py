# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from CLS.FeatureSelection import FeatureSelection


class Cluster(object):
    ##定义全局属性##
    __Colors = ['r', 'y', 'b', 'c', 'm', 'g', 'k', 'w']  # 颜色
    __Markers = ['+', 's', '*', '^', 'x', 'o', 'v', 'x']  # 标记
    __kmean_df = None  # 聚类结果
    __k = None  # 聚类数

    def __init__(self, DataFrame, min_k=3, max_k=9):  ## 1.初始化建立聚类##
        ##特征选择及数据准备
        new_order = np.random.permutation(len(DataFrame))
        new_df = DataFrame.take(new_order)
        f = FeatureSelection(new_df)
        feature_names = f.return_feature_set(variance_threshold=True, tree_select=True)
        feature = new_df[feature_names]
        # K均值聚类
        Kmeans = []
        for k in range(min_k, max_k):
            kmean = KMeans(n_clusters=k, random_state=10).fit(feature)
            Kmeans.append((k, metrics.calinski_harabaz_score(feature, kmean.labels_), kmean))
        Best_kmean = sorted(Kmeans, key=lambda d: d[1])[-1]  # 选择最好聚类
        feature['jllable'] = Best_kmean[-1].labels_
        feature['流失标志'] = new_df['流失标志']
        self.__k = Best_kmean[0]
        self.__kmean_df = feature
        self.Cluster_assess(Kmeans)  # 输出效果
        print('分成了%s个聚类！' % self.__k)

    def Cluster_assess(self, Kmeans):  ## 2.1聚类效果评价##
        plt.figure(figsize=(13, 5))
        plt.subplot(1, 2, 1)
        plt.title('Calinski-Harabasz分数值评估')
        plt.xlabel('K')
        plt.ylabel('Scores')
        plt.plot(np.array(Kmeans)[:, 0], np.array(Kmeans)[:, 1])
        plt.subplot(1, 2, 2)
        feature = self.__kmean_df.drop(['流失标志', 'jllable'], axis=1)
        pca = PCA(n_components=2)
        new_pca = pd.DataFrame(pca.fit_transform(feature))
        plt.title('聚类效果分布散点图降维（二维）')
        for i in range(self.__k):
            d = new_pca[self.__kmean_df['jllable'] == i]
            plt.plot(d[0], d[1], self.__Colors[i] + self.__Markers[i])
        plt.legend(np.char.add('聚类', list(map(str, range(self.__k)))))
        plt.gcf().savefig('Cluster_assess.png')
        plt.show()

    def Cluster_rate(self):  ## 3.1聚类流失率分析##
        plt.figure(figsize=(13, 5))
        Ser_count_type = self.__kmean_df['jllable'].value_counts()
        Ser_runoff_rate = self.__kmean_df['流失标志'].groupby(self.__kmean_df['jllable']).mean()
        df_runoff_rate = pd.concat([Ser_count_type, Ser_runoff_rate], axis=1, keys=['客户数量', '流失率'])
        labels = np.char.add('聚类', list(map(str, df_runoff_rate.index)))
        values_count = df_runoff_rate['客户数量']
        values_rate = df_runoff_rate['流失率']
        plt.subplot(1, 2, 1)
        plt.title('航空客户数量饼图')
        explode = [0] * self.__k
        explode[values_rate.idxmax()] = 0.3
        plt.pie(values_count, labels=labels, colors=self.__Colors[:self.__k], explode=explode, startangle=180,
                shadow=True, autopct='%1.2f%%', pctdistance=0.5, textprops={'fontsize': 14, 'color': 'k'},
                wedgeprops={'linewidth': 1.5, 'edgecolor': 'w'})
        plt.axis('equal')
        plt.subplot(1, 2, 2)
        plt.title('航空客户流失率条形图')
        plt.bar(labels, values_rate, color=self.__Colors[:self.__k])
        plt.grid(True)
        for x, y in zip(labels, values_rate):
            plt.text(x, y, '%.3f' % y, ha='center', va='bottom', fontsize=14)
        plt.gcf().savefig('Cluster_rate.png')
        plt.show()

    def Cluster_cmp(self):  ##3.2聚类特征变量对比（Z标准化）
        new_df = self.__kmean_df.drop(['流失标志', 'jllable'], axis=1).groupby(self.__kmean_df['jllable']).mean()
        result = new_df.copy()

        def f(x):
            return (x - np.mean(x)) / np.std(x)

        new_df = new_df.apply(f).T
        barh = new_df.plot(kind='barh', figsize=(17, 30), fontsize=20, grid=True, color=self.__Colors[:self.__k])
        barh.legend(np.char.add('聚类', list(map(str, new_df.columns))), fontsize=16, loc=2)
        plt.gcf().savefig('Cluster_cmp.png')
        plt.show()
        return result
