# -*- coding: utf-8 -*-
# create by Xu
# date 7/23/2020

import pandas as pd

datafile = r'./air_data.csv'  # 航空原始数据，第一行为属性标签
data = pd.read_csv(datafile, encoding='gb18030')  # 读取原始数据
explore = data.describe(percentiles=[], include='all').T
# 包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）；T是转置，转置后更方便查阅
print(explore)
explore['null'] = len(data) - explore['count']  # describe()函数自动计算非空值数，空值数需手动计算
explore = explore[['null', 'max', 'min', 'mean', 'std']]
explore.columns = ['空值数', '最大值', '最小值', '均值', '方差']  # 表头重命名
print('-----------------------------------------------------------------以下是处理后数据')
print(explore)
'''describe()函数自动计算的字段有count（非空值数）、unique（唯一值数）、top（频数最高者）、freq（最高频数）、
mean（平均值）、std（方差）、min（最小值）、50%（中位数）、max（最大值）'''
