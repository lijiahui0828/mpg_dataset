import numpy as np

class linear_regression:
    """
    the linear regression solver
    """
    def __init__(self, train_data, theta=0):
        """
        initialization of the feature of class
        :param train_data: the train data
        :param theta: the parameter
        """
        self.theta = theta
        self.train_dat = train_data

    def train(self, variable_name, poly_order):
        """
        train the model
        :param variable_name: the variable we use to train the model
        :param poly_order: the polynomial order we choose
        """
        y = self.train_dat.mpg.values
        self.var = variable_name
        self.order = poly_order
        x = self.train_dat[self.var]
        X = np.ones(len(self.train_dat))
        if self.order == 0:
            self.theta = 1 / (X.T @ X) * X.T @ y
        else:
            for i in range(1, self.order + 1):
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
        x_te = test_dat[self.var]
        X_te = np.ones(len(x_te))
        for i in range(1, self.order + 1):
            X_te = np.c_[X_te, x_te ** (i)]
        return (np.dot(X_te, self.theta))