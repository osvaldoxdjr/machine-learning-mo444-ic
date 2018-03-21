import pandas as pd
import numpy as np
import random

def feature_scaling(data, type='std'):

    m_examples = np.shape(data)[0]
    n_features = np.shape(data)[1]
    for n in range(n_features):
        if type == 'minmax':
            min = data.iloc[:,n].min()
            max = data.iloc[:,n].max()
            for m in range(m_examples):
                data.iloc[m,n] = (data.iloc[m,n]-min)/(max-min)
        elif type == 'std':
            mean = data.iloc[:,n].mean()
            std = data.iloc[:,n].std()
            for m in range(m_examples):
                data.iloc[m,n] = (data.iloc[m,n]-mean)/std
    return data

def perform_GD(thetas, LR, feature_train, target_train):
    print("Performing GD!\n")
    new_thetas = []
    residual = []
    m_examples = np.shape(feature_train)[0]
    n_features = np.shape(feature_train)[1]
    sum = 0


    for n in range(n_features):
        if n%10 == 0:
            print("%i thetas calculated!"%n)
        for m in range(m_examples):
            if n == 0:
                res = np.dot(thetas, feature_train.iloc[m,:])-target_train[m]
                residual.append(res)
                sum += res
            else:
                sum += residual[m] * feature_train.iloc[m,n]

        new_thetas.append(thetas[n]-sum*LR/m_examples)
        sum = 0

    print(new_thetas)

    return new_thetas

def cost_function(thetas, data_test, data_target):

    m_examples = np.shape(data_test)[0]
    sum = 0

    for m in range(m_examples):
        sum += (np.dot(thetas, data_test.iloc[m, :]) - data_target[m])**2
    return (sum/2*m_examples)

if __name__ == "__main__":

    LR = 0.1
    t = []
    v = []
    x0 = 1

    data_test = pd.read_csv('test.csv')
    data_test_target = pd.read_csv('test_target.csv')
    data_train = pd.read_csv('train.csv')

    feature_train = data_train.iloc[:1000,2:60]
    target_train = data_train.iloc[:1000,60]

    feature_validation = data_train.iloc[1001:2000,2:60]
    target_validation = data_train.iloc[1001:2000,60]

    feature_validation = feature_validation.reset_index(drop=True)
    target_validation = target_validation.reset_index(drop=True)

    for i in range(np.shape(feature_train)[0]):
        t.append(x0)

    for i in range(np.shape(feature_validation)[0]):
        v.append(x0)

    #feature_test = data_test.iloc[0:100,2:]
    #target_test = data_target.iloc[0:100,0]

    feature_scaling(feature_train, 'minmax')
    feature_scaling(feature_validation, 'minmax')

    #feature_scaling(feature_test)

    #feature_test.insert(0, 'X0', t)

    feature_train.insert(0, 'X0', t)

    feature_validation.insert(0, 'X0', v)

    thetas = np.ones(np.shape(feature_train)[1])


    a_value = 0

    for i in range(100):
        n_value = cost_function(thetas, feature_validation, target_validation)
        print("The cost is: %.6f"%n_value)
        print("The diff from previous and actual is: %f\n"%(n_value-a_value))
        thetas = perform_GD(thetas, LR, feature_train, target_train)
        a_value = n_value