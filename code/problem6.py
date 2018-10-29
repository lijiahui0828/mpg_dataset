import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from basicfunction import read_data, set_var_list, split_data
from problem1 import set_category

def Precision(y_true, y_pred, category):
    """
    define the precision of classification
    :param y_true: the true target value: mpg category
    :param y_pred: the predicted value: mpg category
    :param category: the category that we want to get the classification precision
    :return: the precision
    """
    ind = (y_pred ==category)
    TP = sum(y_true[ind] == y_pred[ind])
    FP = sum(y_true[ind] != y_pred[ind])
    return (TP/ (TP + FP))


if __name__ == "__main__":
    """
    train the logistic regression model and check the precision
    """
    data = read_data()
    _, _, data = set_category(data)
    var_list = set_var_list()
    train_data, test_data = split_data(data)

    df_3 = pd.DataFrame(np.zeros(6).reshape(3, 2))
    df_3.columns = ['train', 'test']
    df_3.index = ['low', 'medium', 'high']

    logistic = LogisticRegression( random_state = 0)
    logistic.fit( train_data[var_list], train_data.category)

    log_pred = logistic.predict(train_data[var_list])
    for cat in df_3.index:
        df_3.iloc[df_3.index == cat, 0] = Precision(train_data.category, log_pred, cat)

    log_pred = logistic.predict(test_data[var_list])
    for cat in df_3.index:
        df_3.iloc[df_3.index == cat, 1] = Precision(test_data.category, log_pred, cat)

    print(df_3)