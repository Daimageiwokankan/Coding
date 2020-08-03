# -*- coding: utf-8 -*-

'''去除方差小的特征'''
from pandas import DataFrame
from sklearn.feature_selection import VarianceThreshold

'''RFE_CV'''
from sklearn.ensemble import ExtraTreesClassifier

'''随机稀疏模型'''
from stability_selection.randomized_lasso import RandomizedLogisticRegression

'''特征选择'''
from sklearn.feature_selection import SelectFromModel


class FeatureSelection(object):
    def __init__(self, DataFrame):
        self.train_test = DataFrame.drop(['会员卡号', '流失标志'], axis=1)  # features #
        self.label = DataFrame['流失标志']  # target   #
        self.feature_name = list(self.train_test.columns)  # feature name #

    def variance_threshold(self):  # 方差选择
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        feature_num = sel.fit_transform(self.train_test, self.label).shape[1]
        feature_var = list(sel.variances_)  # feature variance #
        features = dict(zip(self.feature_name, feature_var))
        features = list(dict(sorted(features.items(), key=lambda d: d[1])).keys())[-feature_num:]
        return set(features)  # return set type #

    def tree_select(self):  # 树模型选择
        clf = ExtraTreesClassifier(n_estimators=300, max_depth=7, n_jobs=4).fit(self.train_test, self.label)
        model = SelectFromModel(clf, prefit=True)  # feature select#
        feature_num = model.transform(self.train_test).shape[1]
        feature_var = list(clf.feature_importances_)  # feature scores #
        features = dict(zip(self.feature_name, feature_var))
        features = list(dict(sorted(features.items(), key=lambda d: d[1])).keys())[-feature_num:]
        return set(features)  # return set type #

    def rlr_select(self):  # 随机稀释模型选择
        clf = RandomizedLogisticRegression()
        feature_num = clf.fit_transform(self.train_test, self.label).shape[1]
        feature_var = list(clf.scores_)  # feature scores #
        features = dict(zip(self.feature_name, feature_var))
        features = list(dict(sorted(features.items(), key=lambda d: d[1])).keys())[-feature_num:]
        return set(features)  # return set type #

    def return_feature_set(self, variance_threshold=False, tree_select=False, rlr_select=False):
        names = set([])
        if variance_threshold is True:
            name_one = self.variance_threshold()
            names = names.union(name_one)
        if tree_select is True:
            name_two = self.tree_select()
            names = names.intersection(name_two)
        if rlr_select is True:
            name_three = self.rlr_select()
            names = names.intersection(name_three)
        return list(names)
