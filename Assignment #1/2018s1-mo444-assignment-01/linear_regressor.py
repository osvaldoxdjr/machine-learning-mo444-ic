import pandas as pd
import numpy as np
import random

def feature_scaling(data, type='std'):

    n_features = np.shape(data)[1]

    for n in range(n_features):
        if type == 'minmax':
            min = data.iloc[:, n].min()
            max = data.iloc[:, n].max()
            data.iloc[:, n] = (data.iloc[:, n] - min) / (max - min)
        elif type == 'std':
            mean = data.iloc[:, n].mean()
            std = data.iloc[:, n].std()
            data.iloc[:, n] = (data.iloc[:, n] - mean) / std
    return data

def perform_GD(thetas, LR, feature_train, target_train):

    print("Performing GD!\n")
    m_examples = np.shape(feature_train)[0]
    error = np.dot(thetas,feature_train.transpose())-target_train
    thetas = thetas - np.dot(feature_train.transpose(),error)*LR/m_examples
    return thetas

def perform_NE(feature_train, target_train):

    a1 = np.linalg.pinv(np.dot(feature_train.transpose(),feature_train))
    thetas = np.dot(a1,feature_train.transpose())
    thetas = np.dot(thetas,target_train)
    return thetas

def cost_function(thetas, data_test, data_target):

    m_examples = np.shape(data_test)[0]
    error = np.square((np.dot(thetas, feature_train.transpose())) - target_train)
    return np.sum(error)/m_examples

if __name__ == "__main__":

    LR = 0.1
    t = []
    v = []
    x0 = 1

    data_test = pd.read_csv('test.csv')
    data_test_target = pd.read_csv('test_target.csv')
    data_train = pd.read_csv('train.csv')

    feature_train = data_train.iloc[:24500,2:60]
    target_train = data_train.iloc[:24500,60]

    feature_validation = data_train.iloc[24501:,2:60]
    target_validation = data_train.iloc[24501:,60]

    feature_validation = feature_validation.reset_index(drop=True)
    target_validation = target_validation.reset_index(drop=True)

    for i in range(np.shape(feature_train)[0]):
        t.append(x0)

    for i in range(np.shape(feature_validation)[0]):
        v.append(x0)

    feature_scaling(feature_train, 'minmax')
    feature_scaling(feature_validation, 'minmax')

    feature_train.insert(0, 'X0', t)

    feature_validation.insert(0, 'X0', v)

    thetas = np.ones(np.shape(feature_train)[1])


    a_value = 0
    diff = 200


    while abs(diff) >= 10:
        n_value = cost_function(thetas, feature_validation, target_validation)
        print("The cost is: %e"%n_value)
        diff = n_value-a_value
        print("The diff from previous and actual is: %e\n"%diff)
        thetas = perform_GD(thetas, LR, feature_train, target_train)
        a_value = n_value
    print(thetas)
    print('separacao')
    a = perform_NE(feature_train,target_train)
    print(a)
    print(cost_function(a,feature_validation,target_validation))
    print(np.shape(a),np.shape(thetas))