# -*- coding: GB2312 -*-
import numpy
import numpy as np
import pandas as pd
from math import log
import operator
from sklearn.metrics import r2_score, mean_squared_error

f = open('Client_Info.csv', encoding='GB2312')
data = pd.read_csv(f)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:, :5], data.iloc
[:, 5], test_size=0.3, shuffle=True)


class Decision_Tree_C45:
    # define entropy calculation
    def __init__(self):
        self.tmp = 0

    def Entropy(self, train_data):
        """
        Parameters
        ----------
        train_data : list
        DESCRIPTION: list of training instances
        Returns
        -------
        ent: entropy of data
        """
        inst_num = len(train_data)  # instances number
        label_counts = {}  # count instances of each class
        for i in range(inst_num):
            label = train_data[i][-1]  # get instance class
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1  # count
        ent = 0
        for key in label_counts.keys():
            # calculate each class proportion
            prob = float(label_counts[key]) / inst_num
            ent -= prob * log(prob, 2)  # see Eq.(3.1)
        return ent

    # split data according to feature and feature value
    def split_data(self, train_data, feature_index, feature_value, feature_type):
        """
        Parameters
        ----------
        train_data : list
        DESCRIPTION: list of training instances
        feature_index : int
        DESCRIPTION: index of feature to split
        feature_value : float,int or str
        DESCRIPTION: feature value
        feature_type : "D",L" or "R"
        DESCRIPTION: "D" for discrete feature,"L" for left split of
        contimuous feature,"R"for left split of contimuous feature
        Returns
        -------
        splitedData: data splits
        """
        splitedData = []  # store splited data
        if feature_type == "D":  # for discrete feature
            for feat_vect in train_data:
                if feat_vect[feature_index] == feature_value:
                    reducedVect = []
                    # delete used discrete feature from data
                    for i in range(len(feat_vect)):
                        if i < feature_index or i > feature_index:
                            reducedVect.append(feat_vect[i])
                    splitedData.append(reducedVect)
        if feature_type == "L":  # for continous feature
            for feat_vect in train_data:
                if feat_vect[feature_index] <= feature_value:
                    splitedData.append(feat_vect)
        if feature_type == "R":  # for continous feature
            for feat_vect in train_data:
                if feat_vect[feature_index] > feature_value:
                    splitedData.append(feat_vect)
        return splitedData

    # choose best feature to split
    def choose_split_feature(self, train_data):
        """
        Parameters
        ----------
        train_data : list
        DESCRIPTION: list of training instances
        Returns
        -------
        best_feat_index: index of chosen feature
        best_feat_value: value of chosen feature
        """
        feat_num = len(train_data[0]) - 1  # get available features
        base_ent = self.Entropy(train_data)
        bestInforGain = 0.0
        best_feat_index = -1
        best_feat_value = 0
        for i in range(feat_num):
            if isinstance(train_data[0][i], str):  # for discrete feature
                feat_list = [example[i] for example in train_data]
                unique_values = set(feat_list)
                newEnt = 0
                for value in unique_values:
                    sub_data = self.split_data(train_data, i, value, "D")
                    prop = float(len(sub_data)) / len(train_data)
                    newEnt += prop * self.Entropy(sub_data)  # see Eq.(3.2)
                inforgain = base_ent - newEnt
                if inforgain > bestInforGain:
                    best_feat_index = i
                    bestInforGain = inforgain
            else:  # for continous feature
                feat_list = [example[i] for example in train_data]
                unique_values = set(feat_list)
                sort_unique_values = sorted(unique_values)
                minEnt = np.inf
                for j in range(len(sort_unique_values) - 1):
                    div_value = (sort_unique_values[j] + sort_unique_values[j + 1]) / 2
                    sub_data_left = self.split_data(train_data, i, div_value, "L")
                    sub_data_right = self.split_data(train_data, i, div_value, "R")
                    prop_left = float(len(sub_data_left)) / len(train_data)
                    prop_right = float(len(sub_data_right)) / len(train_data)
                    ent = prop_left * self.Entropy(sub_data_left) + \
                          prop_right * self.Entropy(sub_data_right)  # see Eq.(3.6)
                    if ent < minEnt:
                        minEnt = ent
                        best_feat_value = div_value
                inforgain = base_ent - minEnt
                if inforgain > bestInforGain:
                    bestInforGain = inforgain
                    best_feat_index = i
        return best_feat_index, best_feat_value

    # get major class
    def get_major_class(self, classList):
        """
        Parameters
        ----------
        classList : TYPE: list
        DESCRIPTION: labels of instances
        Returns
        -------
        major: TYPE: int
        DESCRIPTION: major label
        """
        classcount = {}
        for vote in classList:
            if vote not in classcount.keys():
                classcount[vote] = 0
            classcount[vote] += 1
        sortedclasscount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
        major = sortedclasscount[0][0]
        return major

    # create decision tree
    def create_decision_tree(self, train_data, feat_names, tmp=0):
        classList = [example[-1] for example in train_data]
        # classList = []
        # for x in tmpclassList:
        #     classList.append([x])
        if len(train_data) == 0:  # see condition C
            return
        if len(train_data[0]) == 1:  # see condition B
            return self.get_major_class(classList)
        print(classList.count(classList[0]))
        print(len(classList))
        if classList.count(classList[0]) == len(classList) or classList.count(classList[0])==self.tmp:  # see condition A
            return classList[0]
        self.tmp=classList.count(classList[0])
        # choose best division feature
        best_feat, best_div_value = self.choose_split_feature(train_data)
        if isinstance(train_data[0][best_feat], str):  # for discrete feature
            feat_name = feat_names[best_feat]
            tree_model = {feat_name: {}}  # generate a root node
            del (feat_names[best_feat])  # del feature used
            feat_values = [example[best_feat] for example in train_data]
            unique_feat_values = set(feat_values)
            # create a node for each value of the best feature
            for value in unique_feat_values:
                sub_feat_names = feat_names[:]
                tree_model[feat_name][value] = \
                    self.create_decision_tree(self.split_data(train_data,
                                                              best_feat, value, "D"),
                                              sub_feat_names)
        else:  # for contiunous feature
            print(train_data[0][best_feat])
            best_feat_name = feat_names[best_feat] + "<" + str(best_div_value)
            tree_model = {best_feat_name: {}}  # generate a root node
            sub_feat_names = feat_names
            # generate left node
            tree_model[best_feat_name]["Y"] = \
                self.create_decision_tree(self.split_data(train_data,
                                                          best_feat,
                                                          best_div_value, "L"),
                                          sub_feat_names)
            # generate right node
            tree_model[best_feat_name]["N"] = \
                self.create_decision_tree(self.split_data(train_data,
                                                          best_feat,
                                                          best_div_value, "R"),
                                          sub_feat_names)
        return tree_model

    # define predict function
    def predict(self, tree_model, feat_names, feat_vect):
        """
        Parameters
        ----------
        tree_model : dict
        DESCRIPTION: decision tree model
        feat_names : list
        DESCRIPTION: feature names
        feat_vect :
        DESCRIPTION: feature vector
        Returns
        -------
        label: predicted label
        """
        firstStr = list(tree_model.keys())[0]  # get tree root
        lessIndex = str(firstStr).find('<')
        if lessIndex > -1:  # if root is a continous feature
            # recursively search untill leaft node
            secondDict = tree_model[firstStr]
            feat_name = str(firstStr)[:lessIndex]
            featIndex = feat_names.index(feat_name)
            div_value = float(str(firstStr)[lessIndex + 1:])
            if np.any(feat_vect[featIndex] <= div_value):
                if isinstance(secondDict["Y"], dict):
                    classLabel = self.predict(secondDict["Y"],
                                              feat_names, feat_vect)
                else:
                    classLabel = secondDict["Y"]
            else:
                if isinstance(secondDict["N"], dict):
                    classLabel = self.predict(secondDict["N"],
                                              feat_names, feat_vect)
                else:
                    classLabel = secondDict["N"]
            return classLabel
        else:  # if root is a discrete feature
            # recursively search untill leaft node
            secondDict = tree_model[firstStr]
            featIndex = feat_names.index(firstStr)
            key = feat_vect[featIndex]
            valueOfFeat = secondDict[key]
            if isinstance(valueOfFeat, dict):
                classLabel = self.predict(valueOfFeat, feat_names, feat_vect)
            else:
                classLabel = valueOfFeat
            return classLabel


DC = Decision_Tree_C45()
train_x = np.array(X_train).reshape(-1, 5)
train_y = np.array(Y_train).reshape(-1, 1)
train = numpy.concatenate((train_x, train_y), axis=1)
test_x = np.array(X_test).reshape(-1, 5)
test_y = np.array(Y_test).reshape(-1, 1)
featnames = ['����', '����', '�Ա�', '��ʷ���Ŷ��', '��ʷΥԼ����', '�Ƿ�ΥԼ']
treemodel = DC.create_decision_tree(train, featnames)
# ����Ԥ��
predicted_y = DC.predict(treemodel, featnames, test_x)
# ����ģ�͵�Ԥ������
mse = mean_squared_error(test_y, predicted_y)
r2 = r2_score(test_y, predicted_y)

print("������� (MSE):", mse)
print("ȷ��ϵ�� (R-squared):", r2)

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
mse = mean_squared_error(test_y, predicted_y)
r2 = r2_score(test_y, predicted_y)

print("������� (MSE):", mse)
print("ȷ��ϵ�� (R-squared):", r2)
