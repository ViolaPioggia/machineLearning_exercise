# -*- coding: GB2312 -*-
import numpy
import numpy as np
import pandas as pd
from math import log
import operator
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score

f = open('Client_Info.csv', encoding='GB2312')
data = pd.read_csv(f)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:, :5], data.iloc
[:, 5], test_size=0.3, shuffle=True)


class Decision_Tree_C45:
    def __init__(self):
        self.stump = [0, 0, 0, 0]  # feature_index, value, left_class, right_class

    # ������ũ��
    def cal_Ent(self, dataset, weight):
        sum = np.sum(weight)
        lable_counts = {}  # ��¼ÿ���������
        for i in range(len(dataset)):
            sample_class = dataset[i][-1]
            if sample_class not in list(lable_counts.keys()):
                lable_counts[sample_class] = 0
            lable_counts[sample_class] += weight[i][0]  # ĳ���ĳ��������Ȩ��

        shanonEnt = 0
        for key in lable_counts:
            prob = float(lable_counts[key]) / sum
            shanonEnt -= prob * log(prob, 2)

        return shanonEnt

    # feature_index��ʾʹ���ĸ������ָ����ݼ���value��ʾ������ȡֵ
    def split_dataset(self, dataset, feature_index, value):
        sub_dataset_l = []
        sub_dataset_r = []
        for sample in dataset:
            if sample[feature_index] > value:
                sub_dataset_r.append(sample.copy())
            else:
                sub_dataset_l.append(sample.copy())
        return sub_dataset_l, sub_dataset_r

    def choose_best_feature(self, dataset, weight):
        feature_num = len(dataset[0]) - 1
        dataset_Ent = self.cal_Ent(dataset, weight)
        best_feature = 0
        best_value = 0
        best_info_gain = 0.0
        for i in range(feature_num):  # ������������˳�򣬼��������������Ϣ����
            values = [example[i] for example in dataset]  # ��i������������ȡֵ
            unique_values = list(set(values))  # �����ظ�����
            unique_values.sort()  # ��ȡֵ�Ĵ�С��������
            t_a = [(unique_values[i + 1] + unique_values[i]) / 2 for i in range(len(unique_values) - 1)]  # �����������м�ֵ

            for value in t_a:
                sub_dataset_l, sub_dataset_r = self.split_dataset(dataset, i, value)  # ��ȡ����i����������value���Ӽ�
                sub_Ent = len(sub_dataset_l) / float(len(dataset)) * self.cal_Ent(sub_dataset_l, weight) \
                          + len(sub_dataset_r) / float(len(dataset)) * self.cal_Ent(sub_dataset_r,
                                                                                    weight)  # ÿ���Ӽ��ľ����ؼ�Ȩ��ͣ� ������������

                info_gain = dataset_Ent - sub_Ent  # �������������Ϣ����
                if info_gain > best_info_gain:  # ȡ��Ϣ��������������Ϊ��������
                    best_info_gain = info_gain
                    best_feature = i
                    best_value = value
        return best_feature, best_value

    # �����Ӽ�������������𣬲���������ֵ
    def majority_count(self, classlist, weight):
        classes = {}
        for i in range(len(classlist)):
            if classlist[i] not in list(classes.keys()):
                classes[classlist[i]] = 0
            classes[classlist[i]] += weight[i][0]

        sorted_classes = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)  # ������Ϊ1��ֵ����������
        return sorted_classes[0][0]

    def fit(self, dataset, weight):

        best_feature, best_value = self.choose_best_feature(dataset, weight)
        self.stump[0] = best_feature
        self.stump[1] = best_value

        sub_dataset_l, sub_dataset_r = self.split_dataset(dataset, best_feature, best_value)

        classlist_l = [sample[-1] for sample in sub_dataset_l]
        self.stump[2] = self.majority_count(classlist_l, weight)

        classlist_r = [sample[-1] for sample in sub_dataset_r]
        self.stump[3] = self.majority_count(classlist_r, weight)

    def predict(self, x):
        feature_index = x[:, self.stump[0]].reshape([-1, 1])
        pred = np.where(feature_index < self.stump[1], self.stump[2], self.stump[3])

        return pred


train_x = np.array(X_train).reshape(-1, 5)
train_y = np.array(Y_train).reshape(-1, 1)
train_dataset = numpy.concatenate((train_x, train_y), axis=1)
test_x = np.array(X_test).reshape(-1, 5)
test_y = np.array(Y_test).reshape(-1, 1)
featnames = ['����', '����', '�Ա�', '��ʷ���Ŷ��', '��ʷΥԼ����']
weight = np.ones([1000, 1])

DC = Decision_Tree_C45()
DC.fit(train_dataset, weight)
predicted_y = DC.predict(test_x)
auc = accuracy_score(test_y, predicted_y)
pre = precision_score(test_y, predicted_y)

print("auc=", auc)
print("pre=", pre)

from sklearn.tree import DecisionTreeClassifier

DC = DecisionTreeClassifier()
train_x = np.array(X_train).reshape(-1, 5)
train_y = np.array(Y_train).reshape(-1, 1)
test_x = np.array(X_test).reshape(-1, 5)
test_y = np.array(Y_test).reshape(-1, 1)
train_yy = train_y.flatten()
DC.fit(train_x, train_yy)
# ����Ԥ��
predicted_y = DC.predict(test_x)
# ����ģ�͵�Ԥ������
auc = accuracy_score(test_y, predicted_y)
pre = precision_score(test_y, predicted_y)

print("auc=", auc)
print("pre=", pre)
