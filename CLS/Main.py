from pandas import DataFrame
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# new_df = df
new_df = DataFrame
feature = new_df[['会员卡级别', '总票价', '总累计积分']]
# 初步聚类
kmean = KMeans(n_clusters=3, random_state=10).fit(feature)
new_df['jllable'] = kmean.labels_
# 绘制图表比较
Mean = feature.groupby(kmean.labels_).mean()
figsize = 11, 12
index = Mean.index
colors = ['r', 'y', 'b']
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=figsize, facecolor='w', edgecolor='blue', sharex=True,
                       sharey='row')  # 返回figure、axes对象
# 会员卡级别
ax[0].bar(index, Mean['会员卡级别'], color=colors)
ax[0].set_title('会员卡级别', fontstyle='italic', fontsize=12)
for x, y in zip(index, Mean['会员卡级别']):
    ax[0].text(x, y, '%.2f' % y, ha='center', va='bottom', fontsize=12)
# 总票价
ax[1].bar(index, Mean['总票价'], color=colors)
ax[1].set_title('总票价', fontstyle='italic', fontsize=12)
for x, y in zip(index, Mean['总票价']):
    ax[1].text(x, y, '%.2f' % y, ha='center', va='bottom', fontsize=12)
# 总积分
ax[2].bar(index, Mean['总累计积分'], color=colors)
ax[2].set_title('总累计积分', fontstyle='italic', fontsize=12)
for x, y in zip(index, Mean['总累计积分']):
    ax[2].text(x, y, '%.2f' % y, ha='center', va='bottom', fontsize=12)
plt.xticks(index, ['聚类0', '聚类1', '聚类2'], fontsize=16)
plt.gcf().savefig('Cluster_select.png')
plt.show()
