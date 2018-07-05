import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y,y_predict):
    assert y.shape == y_predict.shape, \
        "the size of y must be equal to the size of the y_predict"

    return sum((y - y_predict)**2)/len(y)

def root_mean_squared_error(y,y_predict):

    return sqrt(mean_squared_error(y,y_predict))

def mean_absolute_error(y,y_predict):
    assert y.shape == y_predict.shape, \
        "the size of y must be equal to the size of the y_predict"

    return sum(np.absolute(y - y_predict))/len(y)

def r2_score(y,y_predict):

    return 1-(mean_squared_error(y,y_predict)/np.var(y))


def train_test_split(X,y,test_ratio=0.2,seed = None):
    '''
    :param X: input data set
    :param y: input lable set
    :param test_ratio: the proportion of test data set
    :param seed: random seed
    :return: X_train,y_train,X_test,y_test
    '''

    assert X.shape[0] == y.shape[0], \
        "the len of the X must be equal to the len the y"
    assert 0 <= test_ratio <= 1, \
        "test_ratio must be more than 0 and less than 1"
    if seed:
        np.random.seed(seed)

    # 打乱数据
    shuffle_indexes = np.random.permutation(len(X))
    test_number = int(len(X) * test_ratio)
    X_test = X[shuffle_indexes[:test_number], :]
    y_test = y[shuffle_indexes[:test_number]]
    X_train = X[shuffle_indexes[test_number:], :]
    y_train = y[shuffle_indexes[test_number:]]

    return X_train,X_test,y_train,y_test


