import pandas as pd
import numpy as np
from basicfunction import read_data, set_var_list, split_data
from problem1 import set_category
from problem5 import linear_regression_2
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    """
    train the model with whole data and predict the value of the new data
    """
    data = read_data()
    _, _, data = set_category(data)
    var_list = set_var_list()

    lin_3 = linear_regression_2(data)
    lin_3.train(var_list, 2)
    new = pd.DataFrame({'cylinders':6, 'displacement':350, 'horsepower': 180, 'weight':3700, 'acceleration':9,
                        'modelyear':80,'origin': 1}, index=[0])
    print("Polynomial Regression: {}.".format(lin_3.predict(new)[0]))

    logistic_2 = LogisticRegression( random_state = 0)
    logistic_2.fit(data[var_list], data.category)

    new = np.array([6, 350, 180, 3700, 9, 80,1]).reshape(1, -1)
    print("Logistic Regression: {}.".format(logistic_2.predict(new)[0]))