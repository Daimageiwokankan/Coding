# -*- coding: utf-8 -*-
# create by Xu
# date 7/24/2020


# K-Means聚类算法
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  # 导入K均值聚类算法
import matplotlib.pyplot as plt

k = 5  # 需要进行的聚类类别数

# 读取数据并进行聚类分析
data = pd.read_csv('./data/data_standard.csv')
data = data.drop(labels='ZMEMBER_NO', axis=1)
data = data.drop(labels='K', axis=1)

# 调用k-means算法，进行聚类分析
kmodel = KMeans(n_clusters=k, n_jobs=4)  # n_jobs是并行数，一般等于CPU数较好
kmodel.fit(data)  # 训练模型

# kmodel.cluster_centers_ #查看聚类中心
# kmodel.labels_ #查看各样本对应的类别

# 简单打印结果
if k == 5:
    s = pd.Series([u'CUS1', u'CUS2', u'CUS3', u'CUS4', u'CUS5'], index=[0, 1, 2, 3, 4])  # 创建一个序列s
elif k == 4:
    s = pd.Series(['CUS1', 'CUS2', 'CUS3', 'CUS4'], index=[0, 1, 2, 3])  # 创建一个序列s
elif k == 3:
    s = pd.Series(['CUS1', 'CUS2', 'CUS3'], index=[0, 1, 2])  # 创建一个序列s
r1 = pd.Series(kmodel.labels_).value_counts()  # 统计各个类别的数目
r2 = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚类中心
r = pd.concat([s, r1, r2], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = [u'聚类名称'] + [u'聚类个数'] + list(data.columns)  # 重命名表头
print(r)


def plot_radar(data):
    """
    the first column of the data is the cluster name;
    the second column is the number of each cluster;
    the last are those to describe the center of each cluster.
    """
    kinds = data.iloc[:, 0]
    labels = data.iloc[:, 2:].columns
    centers = pd.concat([data.iloc[:, 2:], data.iloc[:, 2]], axis=1)
    centers = np.array(centers)
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)  # 设置坐标为极坐标

    # 画若干个五边形
    floor = np.floor(centers.min())  # 大于最小值的最大整数
    ceil = np.ceil(centers.max())  # 小于最大值的最小整数
    for i in np.arange(floor, ceil + 0.5, 0.5):
        ax.plot(angles, [i] * (n + 1), '--', lw=0.5, color='black')

    # 画不同客户群的分割线
    for i in range(n):
        ax.plot([angles[i], angles[i]], [floor, ceil], '--', lw=0.5, color='black')

    # 画不同的客户群所占的大小
    for i in range(len(kinds)):
        ax.plot(angles, centers[i], lw=2, label=kinds[i])
        # ax.fill(angles, centers[i])

    ax.set_thetagrids(angles * 180 / np.pi, labels)  # 设置显示的角度，将弧度转换为角度
    plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0.0))  # 设置图例的位置，在画布外

    ax.set_theta_zero_location('N')  # 设置极坐标的起点（即0°）在正北方向，即相当于坐标轴逆时针旋转90°
    ax.spines['polar'].set_visible(False)  # 不显示极坐标最外圈的圆
    ax.grid(False)  # 不显示默认的分割线
    ax.set_yticks([])  # 不显示坐标间隔

    plt.show()


plot_radar(r)  # 调用雷达图作图函数
