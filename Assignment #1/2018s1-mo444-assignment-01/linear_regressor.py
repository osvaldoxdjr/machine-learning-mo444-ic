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

def remove_discrete_variables(feature_train):
    discrete_index = []

    for i in range(np.shape(feature_train)[1]):
        if len(np.unique(feature_train[:,i])) == 2:
            discrete_index.append(i)
    return np.delete(feature_train, discrete_index, 1)

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

def plot_hist(y):
    plt.hist(y, 1000)
    plt.show()

def removing_examples(feature_train, target_train, threshold):
    outlier_index = []

    for i in range(np.shape(feature_train)[0]):
        if target_train[i] >=  threshold:
            outlier_index.append(i)

    return np.delete(feature_train, outlier_index, 0), np.delete(target_train, outlier_index, 0)

def plot_function_3d(y, x, ne):
    pass

if __name__ == "__main__":

    LR = 0.01

    # Load data
    data_test = pd.read_csv('test.csv')
    data_test_target = pd.read_csv('test_target.csv')
    data_train = pd.read_csv('train.csv')

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

    # Remove discrete variables
    #feature_train = remove_discrete_variables(feature_train)

    # Inserting X0 to feature datasets
    np.insert(feature_train, 0, tr)
    np.insert(feature_validation, 0, va)
    np.insert(feature_test, 0, te)

    #feature_train = increase_complexity(feature_train,1)

    target_train = np.asarray(target_train)

    plot_hist(target_train)

    m_before = np.shape(feature_train)[0]

    feature_train, target_train = removing_examples(feature_train, target_train, 5000)

    m_after = np.shape(feature_train)[0]

    print('Foram removidos %i exemplos'%(m_before-m_after))

    plot_hist(np.asarray(target_train))

    aux = feature_train.copy()
    J = []


    #for i in range(np.shape(feature_train)[1]-1):
    for i in range(1):
        #feature_train = np.delete(feature_train, i+1, 1)

        # Initializing thetas
        thetas = np.ones(np.shape(feature_train)[1])

        a_value = 0

        x = []
        y = []

        n = 1

        # Perfoming GD and calulating cost function
        for i in range(100):
            n_value = cost_function(thetas, feature_train, target_train)
            y.append(n_value)
            x.append(n)
            n += 1
            print("The cost is: %e"%n_value)
            diff = n_value-a_value
            print("The diff from previous and actual is: %e\n"%diff)
            pred = (np.dot(thetas, feature_train.transpose()))
            #print('Custo R2: %f'%r2_score(target_train, pred))
            thetas = perform_GD(thetas, LR, feature_train, target_train)
            a_value = n_value
        print(thetas)
        print('separacao')
        a = perform_NE(feature_train, target_train)
        print(a)
        print('%e'%(cost_function(a,feature_train,target_train)))
        print('%e' % (cost_function(thetas, feature_train, target_train)))

        print(np.shape(a),np.shape(thetas))

        plot_function_2d(y,x, np.full(len(x),cost_function(a,feature_train,target_train)))

        J.append(n_value)
        feature_train = aux.copy()

    for j in J:
        print('%e'%j)

    target_train = np.asarray(target_train)

    for i in range(20):
        print('%.2f - %.2f'%(np.dot(thetas, feature_train.transpose())[i], target_train[i]))

    mean_err = abs((target_train - np.dot(thetas, feature_train.transpose()))).mean()

    print("A média de erro do número de compartilhamentos é %.2f"%mean_err)

    print('\n\n\n')

    target_test = np.asarray(target_test)
    feature_test = np.asarray(feature_test)

    #print(cost_function(thetas, feature_test, target_test))

    for i in range(20):
        print('%.2f - %.2f'%(np.dot(thetas, feature_test.transpose())[i], target_test[i]))

    mean_err = abs((target_test - np.dot(thetas, feature_test.transpose()))).mean()

    print("A média de erro do número de compartilhamentos é %.2f"%mean_err)