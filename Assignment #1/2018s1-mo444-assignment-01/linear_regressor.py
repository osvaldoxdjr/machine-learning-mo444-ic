import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import random

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
    error = np.square((np.dot(thetas, data_test.transpose())) - data_target)
    return np.sum(error)/(2*m_examples)

def increase_complexity(feature_train, comp):

    n_features = np.shape(feature_train)[1]

    aux = feature_train

    for n in range(n_features):
        m = np.multiply(feature_train[:,n],aux[:,n:].transpose())
        feature_train = np.concatenate((feature_train, m.transpose()), axis=1)
    return feature_train

def plot_function_2d(y, x, ne):

    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()
    ax.plot(x, y, 'k--', label='Gradient Discent')
    ax.plot(x, ne, 'k:', label='Normal Equation')

    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#00FFCC')

    plt.show()

def plot_function_3d(y, x, ne):
    pass

if __name__ == "__main__":

    LR = 0.01

    # Load data
    data_test = pd.read_csv('test.csv')
    data_test_target = pd.read_csv('test_target.csv')
    data_train = pd.read_csv('train.csv')

    #for i in range(np.shape(data_train)

    # Select train predictable features and target
    feature_train = data_train.iloc[:,2:60]
    target_train = data_train.iloc[:,60]

    # Select validation predictable features and target
    feature_validation = data_train.iloc[24501:,2:60]
    target_validation = data_train.iloc[24501:,60]

    # Select test predictable features and target
    feature_test = data_test.iloc[:,2:]
    target_test = data_test_target

    # Adjust index after iloc
    feature_validation = feature_validation.reset_index(drop=True)
    target_validation = target_validation.reset_index(drop=True)

    # Generating X0 = 1
    tr = np.ones(np.shape(feature_train)[0])
    va = np.ones(np.shape(feature_validation)[0])
    te = np.ones(np.shape(feature_test)[0])

    # Feature Scalling
    feature_train  = MinMaxScaler().fit_transform(feature_train)
    feature_validation = MinMaxScaler().fit_transform(feature_validation)

    #target_train = target_train.astype('float64')
    feature_test = np.asarray(feature_test)

    #target normalization
    #target_train = (target_train-target_train.min())/(target_train.max()-target_train.min())

    # Inserting X0 to feature datasets
    np.insert(feature_train, 0, tr)
    np.insert(feature_validation, 0, va)
    np.insert(feature_test, 0, te)

    feature_train = increase_complexity(feature_train,1)

    # Initializing thetas
    thetas = np.ones(np.shape(feature_train)[1])

    a_value = 0
    diff = 200

    x = []
    y = []

    n = 1


    # Perfoming GD and calulating cost function
    while abs(diff) >= 20:
        n_value = cost_function(thetas, feature_train, target_train)
        y.append(n_value)
        x.append(n)
        n += 1
        print("The cost is: %e"%n_value)
        diff = n_value-a_value
        print("The diff from previous and actual is: %e\n"%diff)
        pred = (np.dot(thetas, feature_train.transpose()))
        print('Custo R2: %f'%r2_score(target_train, pred))
        thetas = perform_GD(thetas, LR, feature_train, target_train)
        a_value = n_value
    print(thetas)
    print('separacao')
    a = perform_NE(feature_train, target_train)
    print(a)
    print('%e'%(cost_function(a,feature_train,target_train)))
    print('%e' % (cost_function(thetas, feature_train, target_train)))

    print(np.shape(a),np.shape(thetas))

    print('Y predicted')
    print((np.dot(thetas,feature_validation.transpose()))*(target_train.max()-target_train.min())+target_train.min())
    print(target_validation)

    plot_function_2d(y,x, np.full(len(x),cost_function(a,feature_train,target_train)))