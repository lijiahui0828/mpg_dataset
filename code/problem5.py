import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from basicfunction import read_data, set_var_list, split_data, mse

class linear_regression_2:
    """
    modify the linear regression solver to train all variable
    """

    def __init__(self, train_data, theta = 0):
        """
        initialization of the feature of class
        :param train_data: the train data
        :param theta: the parameter
        """
        self.theta = theta
        self.train_dat = train_data

    def train(self ,variable_name, poly_order):
        """
        train the model
        :param variable_name: the variable we use to train the model
        :param poly_order: the polynomial order we choose
        """
        y = self.train_dat.mpg.values
        self.var = variable_name
        self.order = poly_order
        X = np.ones(len(self.train_dat))
        if self.order == 0:
            self.theta = 1/ (X.T @ X) * X.T @ y
        else:
            for i in range(1, self.order + 1):
                for v in self.var:
                    x = self.train_dat[v]
                    X = np.c_[X, x ** (i)]
            if np.linalg.det(X.T @ X) > 1e-10:
                self.theta = np.linalg.inv(X.T @ X) @ X.T @ y
            else:
                self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, test_dat):
        """
        predict the data by the trained model
        :param test_dat: the test data
        :return: the predicted mpg
        """
        X_te = np.ones(len(test_dat))
        for i in range(1, self.order + 1):
            for v in self.var:
                x_te = test_dat[v]
                X_te = np.c_[X_te, x_te ** (i)]
        return (np.dot(X_te, self.theta))


if __name__ == "__main__":
    """
     train the model with modified linear regression solver and get the result of problem5.
    """
    data = read_data()
    var_list = set_var_list()
    train_data, test_data = split_data(data)

    lin_2 = linear_regression_2(train_data)

    df_2 = pd.DataFrame(np.zeros(6).reshape(3, 2))
    df_2.columns = ['train', 'test']
    df_2.index = ['0th', '1st', '2nd']

    for i in range(3):
        lin_2.train(var_list, i)
        y_pred = lin_2.predict(train_data)
        df_2.iloc[i, 0] = mse(train_data.mpg, y_pred)
        y_pred = lin_2.predict(test_data)
        df_2.iloc[i, 1] = mse(test_data.mpg, y_pred)

    print(df_2)