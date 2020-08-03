# -*- coding: utf-8 -*-
# create by Xu
# date 7/23/2020

import pandas as pd


def clean(fileName):
    cleanedFile = r'./data/data_cleaned.csv'
    data = pd.read_csv(fileName, encoding='gb18030')  # 读取原始数据
    data = data[data['EXPENSE_SUM_YR_1'].notnull() & data['EXPENSE_SUM_YR_2'].notnull()]  # 票价非空值才保留
    # 只保留票价非零的，或者平均折扣率与总飞行公里数同时为0的记录。
    index1 = data['EXPENSE_SUM_YR_1'] != 0
    index2 = data['EXPENSE_SUM_YR_2'] != 0
    index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)  # 该规则是“与”
    index4 = (data['GENDER'] == 0) & (data['age'] == 0)
    data = data[index1 | index2 | index3 | index4]  # 该规则是“或”
    data.to_csv(cleanedFile, encoding='gb18030')
    print(data)


def change(fileName):
    changedFile = r'./data/data_changed.csv'
    data = pd.read_csv(fileName, encoding='gb18030')
    data = data[['MEMBER_NO', 'FFP_days', 'FFP_TIER', 'DAYS_FROM_LAST_TO_END', 'FLIGHT_COUNT', 'avg_discount',
                 'SEG_KM_SUM', 'P1Y_Flight_Count', 'L1Y_Flight_Count', 'runoff_flag']]
    data.to_csv(changedFile, encoding='gb18030')


def LRFMCK(fileName):
    data = pd.read_csv(fileName, encoding='gb18030')
    # 其中K为标签标示用户类型
    data2 = pd.DataFrame(columns=['MEMBER_NO', 'L', 'R', 'F', 'M', 'C', 'K', 'runoff_flag'])
    data2['MEMBER_NO'] = data['MEMBER_NO']
    data2['L'] = data['FFP_days']
    data2['R'] = data['DAYS_FROM_LAST_TO_END']
    data2['F'] = data['FLIGHT_COUNT']
    data2['M'] = data['SEG_KM_SUM']
    data2['C'] = data['avg_discount']
    data2['runoff_flag'] = data['runoff_flag']
    temp = data['L1Y_Flight_Count'] / data['P1Y_Flight_Count']

    for i in range(len(temp)):
        if temp[i] >= 0.9:
            # 未流失客户
            temp[i] = '1'
        elif 0.5 < temp[i] < 0.9:
            # 准流失客户
            temp[i] = '1'
        else:
            # 已流失客户
            temp[i] = '0'

    data2['K'] = temp
    data2.to_csv('./data/data_changed2.csv', encoding='gb18030')


def standard():
    data = pd.read_csv('./data/data_changed2.csv', encoding='gb18030').iloc[:, 1:7]
    zscoredfile = r'./data/data_standard.csv'
    # 简洁的语句实现了标准化变换，类似地可以实现任何想要的变换
    data = (data - data.mean(axis=0)) / (data.std(axis=0))
    data.columns = ['Z' + i for i in data.columns]
    data2 = pd.read_csv('./data/data_changed2.csv', encoding='gb18030')
    data['ZMEMBER_NO'] = data2['MEMBER_NO']
    data['K'] = data2['K']
    data['runoff_flag'] = data2['runoff_flag']
    data.to_csv(zscoredfile, index=False, encoding='gb18030')


if __name__ == '__main__':
    dataFile = r'./data/air_data.csv'
    clean(dataFile)

    dataFile = r'./data/data_cleaned.csv'
    change(dataFile)

    dataFile = r'./data/data_changed.csv'
    LRFMCK(dataFile)
    standard()
