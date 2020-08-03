# -*- coding: utf-8 -*-
# create by Xu
# date 7/23/2020

import pandas as pd

# 丢弃票价为空记录
# 丢弃票价为0、平均折扣率不为0、总飞行公里数大于0的记录

datafile = r'./air_data.csv'  # 航空原始数据,第一行为属性标签
cleanedfile = r'D:/Software/PyCharm/Coding/pre_air_data.csv'  # 数据清洗后保存的文件
data = pd.read_csv(datafile, encoding='gb18030')  # 读取原始数据
data = data[data['EXPENSE_SUM_YR_1'].notnull() & data['EXPENSE_SUM_YR_2'].notnull()]  # 票价非空值才保留
# 只保留票价非零的，或者平均折扣率与总飞行公里数同时为0的记录。
index1 = data['EXPENSE_SUM_YR_1'] != 0
index2 = data['EXPENSE_SUM_YR_2'] != 0
index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)  # 该规则是“与”
data = data[index1 | index2 | index3]  # 该规则是“或”
print(data)
#data.to_csv(cleanedfile)  # 导出结果
