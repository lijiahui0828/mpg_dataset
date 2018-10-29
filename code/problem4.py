import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from basicfunction import read_data, set_var_list, split_data, mse
from problem3 import linear_regression


if __name__ == "__main__":
    """
    train the model with linear regression solver and get the result of problem4: mse and the plot.
    """
    data = read_data()
    var_list = set_var_list()
    train_data, test_data = split_data(data)

    df = pd.DataFrame(np.zeros(56).reshape(7, 8))
    df.columns = ['train_0th', 'test_0th', 'train_1st', 'test_1st',
                  'train_2nd', 'test_2nd', 'train_3rd', 'test_3rd']
    df.index = var_list

    lin = linear_regression(train_data)

    test_pred = []
    for var in var_list:
        test_pred_ = []
        for i in range(4):
            lin.train(var, i)
            y_pred = lin.predict(train_data)
            df.iloc[df.index == var, i * 2] = mse(train_data.mpg, y_pred)
            y_pred = lin.predict(test_data)
            df.iloc[df.index == var, i * 2 + 1] = mse(test_data.mpg, y_pred)
            x_ = {var: np.arange(test_data[var].min(), test_data[var].max(), 0.1)}
            test_pred_.append(lin.predict(x_))
        test_pred.append(test_pred_)

    print(df)

    plt.figure(figsize=(25, 30))
    for i in range(7):
        x = test_data[var_list[i]]
        x_ = np.arange(x.min(), x.max(), 0.1)
        plt.subplot(4, 2, i + 1)
        plt.title('linear regression of mpg vs ' + var_list[i], size=20)
        plt.ylabel('mpg', size=15)
        plt.xlabel(var_list[i], size=15)
        plt.scatter(x, test_data.mpg)
        for j in range(4):
            plt.plot(x_, test_pred[i][j])
    plt.show()



