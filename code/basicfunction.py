import pandas as pd
import numpy as np

def read_data():
    """
    read the data, change the types of some variables, drop the missing value.
    :return data: the data after prepossessing.
    """
    data = pd.read_table("./auto-mpg.data", delim_whitespace=True, header=None)
    data.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'modelyear', 'origin',
                    'carname']
    # remove 6 records with missing values
    data = data.where(data != '?').dropna().reset_index(drop = True)
    data.horsepower = data.horsepower.astype('float')
    return data


def set_var_list():
    """
    get the list of variable which use to build the model.
    :return: the list of variables which use to build the model.
    """
    return ['cylinders', 'displacement', 'horsepower', 'weight','acceleration', 'modelyear', 'origin']

def split_data(data):
    """
    split the data into training set and testing set.
    :param data: the data that need to split (DataFrame).
    :return train_data: the training set.
    :return test_data: the testing set.
    """
    train_data = data.iloc[:200, :]
    test_data = data.iloc[200:, :]
    return train_data, test_data

def mse(y_true, y_pred):
    """
    calculate the mean squared errors.
    :param y_true: the true target value
    :param y_pred: the predicted target value
    :return: the mean squared errors
    """
    return np.mean((y_true - y_pred)**2)