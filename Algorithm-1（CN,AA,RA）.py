# -*- coding=utf-8 -*-


import numpy as np
import pandas as pd
from scipy.special import comb, perm


def cn_sim(x, y):

    index_x = np.where(x != 0)[0]
    index_y = np.where(y != 0)[0]

    cnt = 0
    for key in index_x:
        if key in index_y:
            cnt += 1

    return cnt


def CN(data):

    train_length = int(len(data) * 0.9)
    train_set = data[:train_length, :train_length]
    sim_matrix = np.zeros(train_set.shape)
    m, n = train_set.shape
    sima_mean = 0
    for i in range(m):
        for j in range(i + 1, n):
            tmp = cn_sim(train_set[i], train_set[j])
            sim_matrix[i, j] = tmp
            sima_mean += tmp
            sim_matrix[j, i] = tmp

    sima_mean = sima_mean / perm(m, 2)

    test_set = data[train_length:, :train_length]

    test_sim_value = np.zeros(((len(data) - train_length), train_length))

    for ii in range(len(test_set)):
        for jj in range(train_length):
            test_sim_value[ii, jj] = cn_sim(test_set[ii], train_set[jj])

    AUC_value = AUC_Calc(test_sim_value, test_set, sima_mean)
    Precision_value = Precision_Calc(test_sim_value, test_set, sima_mean)

    print("【CN】:,AUC={},Precision={}".format(AUC_value, Precision_value))


def aa_sim(x, y, data):

    index_x = np.where(x != 0)[0]
    index_y = np.where(y != 0)[0]

    commonpoint_list = []
    for key in index_x:
        if key in index_y:
            commonpoint_list.append(key)
    sum_value = 0
    for key in commonpoint_list:
        point_length = len(np.where(data[key] != 0)[0])
        #value = 1 / np.log(point_length)
        value = 1 / np.log(point_length + 1e-6)
        sum_value += value

    return sum_value


def AA(data):
    train_length = int(len(data) * 0.9)
    train_set = data[(len(data)-train_length):, :train_length]
    sim_matrix = np.zeros(train_set.shape)
    m, n = train_set.shape
    sima_mean = 0
    for i in range(m):
        for j in range(i + 1, n):
            tmp = aa_sim(train_set[i], train_set[j], train_set)
            sim_matrix[i, j] = tmp
            sima_mean += tmp
            sim_matrix[j, i] = tmp

    sima_mean = sima_mean / perm(m, 2)

    test_set = data[:(len(data)-train_length), :train_length]

    test_sim_value = np.zeros(((len(data) - train_length), train_length))

    for ii in range(len(test_set)):
        for jj in range(train_length):
            test_sim_value[ii, jj] = aa_sim(test_set[ii], train_set[jj],train_set)

    AUC_value = AUC_Calc(test_sim_value, test_set, sima_mean)
    Precision_value = Precision_Calc(test_sim_value, test_set, sima_mean)

    print("【AA】:,AUC={},Precision={}".format(AUC_value, Precision_value))


def ra_sim(x, y, data):

    index_x = np.where(x != 0)[0]
    index_y = np.where(y != 0)[0]

    commonpoint_list = []
    for key in index_x:
        if key in index_y:
            commonpoint_list.append(key)
    sum_value = 0
    for key in commonpoint_list:
        point_length = len(np.where(data[key] != 0)[0])
        value = 1 / (point_length + 1e-6)
        sum_value += value

    return sum_value


def RA(data):
    train_length = int(len(data) * 0.9)
    train_set = data[(len(data) - train_length):, :train_length]
    sim_matrix = np.zeros(train_set.shape)
    m, n = train_set.shape
    sima_mean = 0
    for i in range(m):
        for j in range(i + 1, n):
            tmp = ra_sim(train_set[i], train_set[j], train_set)
            sim_matrix[i, j] = tmp
            sima_mean+=tmp
            sim_matrix[j, i] = tmp

    sima_mean = sima_mean / perm(m, 2)

    test_set = data[:(len(data) - train_length), :train_length]

    test_sim_value = np.zeros(((len(data) - train_length), train_length))

    for ii in range(len(test_set)):
        for jj in range(train_length):
            test_sim_value[ii, jj] = ra_sim(test_set[ii], train_set[jj], train_set)

    AUC_value = AUC_Calc(test_sim_value, test_set, sima_mean)
    Precision_value = Precision_Calc(test_sim_value, test_set, sima_mean)

    print("【RA】:,AUC={},Precision={}".format(AUC_value, Precision_value))


def nraw_sim(x,y,data):

    index_x = np.where(x != 0)[0]
    index_y = np.where(y != 0)[0]

    commonpoint_list = []
    for key in index_x:
        if key in index_y:
            commonpoint_list.append(key)

    sum_value = 0
    for key in commonpoint_list:
        point_length = len(np.where(data[key] != 0)[0])

        index_tmp=np.where(data[key]!=0)[0]

        length_inter=0
        length_only=0

        for aa in index_tmp:
            if aa in index_x and aa in index_y:
                length_inter+=1
            elif aa in index_x and aa not in index_y:
                length_only+=1
            elif aa not in index_x and aa in index_y:
                length_only+=1
        value = (1.5*length_inter+0.5*length_only+2) / (point_length + 1e-6)
        sum_value += value

    return sum_value



def NRWA(data):
    train_length = int(len(data) * 0.9)
    train_set = data[(len(data) - train_length):, :train_length]
    sim_matrix = np.zeros(train_set.shape)
    m, n = train_set.shape
    sima_mean = 0
    for i in range(m):
        for j in range(i + 1, n):
            tmp = nraw_sim(train_set[i], train_set[j], train_set)
            sim_matrix[i, j] = tmp
            sima_mean += tmp
            sim_matrix[j, i] = tmp


    sima_mean = sima_mean / perm(m, 2)

    test_set = data[:(len(data) - train_length), :train_length]

    test_sim_value = np.zeros(((len(data) - train_length), train_length))

    for ii in range(len(test_set)):
        for jj in range(train_length):
            test_sim_value[ii, jj] = nraw_sim(test_set[ii], train_set[jj], train_set)

    AUC_value = AUC_Calc(test_sim_value, test_set, sima_mean)
    Precision_value = Precision_Calc(test_sim_value, test_set, sima_mean)

    print("【NRWA 】:,AUC={},Precision={}".format(AUC_value, Precision_value))

def AUC_Calc(testValue, testData, meanValue):
    auc_value=0
    valuess=np.zeros(shape=testValue.shape)
    for i in range(len(testValue)):
        for j in range(len(testValue[0])):
            if testValue[i,j]>=meanValue:
                valuess[i,j]=1.0
            else:
                valuess[i,j]=0.5

    auc_value=np.sum(valuess)/(testValue.shape[0]*testValue.shape[1])

    return auc_value


# 计算precision
def Precision_Calc(testValue, testData, meanValue):
    valuess = np.zeros(shape=testValue.shape)
    for i in range(len(testValue)):
        for j in range(len(testValue[0])):
            if testValue[i, j] >= meanValue and testData[i,j]==1:
                valuess[i, j] = 1
            elif testValue[i, j] <= meanValue and testData[i,j]==0:
                valuess[i, j] = 1

    precision_value = np.sum(valuess) / (testValue.shape[0] * testValue.shape[1])
    return precision_value


if __name__ == '__main__':
    # raw_data = np.array([[0, 1, 1, 0, 0],
    #                      [1, 0, 1, 1, 0],
    #                      [1, 1, 0, 1, 1],
    #                      [0, 1, 1, 0, 1],
    #                      [0, 0, 1, 1, 0]])

    filepath = r"data/data.npy"
    raw_data = np.load(filepath)
    # print(df.shape)

    filepath_2= r"data/cos-adj100.csv"

    df=pd.read_csv(filepath_2,header=None)
    data=df.values
    m,n=data.shape
    for i in range(m):
        for j in range(n):
            if data[i,j]>0.5:
                data[i,j]=1
            else:
                data[i,j]=0

    print(data.shape)
    raw_data=data
    print("************")
    print(raw_data)
    print("******************CN******************")
    sim = CN(raw_data)
    print("******************AA******************")
    sim = AA(raw_data)
    print("******************RA******************")
    sim = RA(raw_data)


