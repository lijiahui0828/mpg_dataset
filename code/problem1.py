import warnings
warnings.filterwarnings('ignore')

from basicfunction import read_data

def set_category(data):
    """
    find the threshold for each mpg category to divided the sample  and set the value of category
    :param data: the data need to set category.
    :return q1: the threshold to divide low and medium mpg.
    :return q2: the threshold to divide medium and high mpg.
    :return data: the data with a new column "category" to show the mpg category.
    """
    q1 = round(data.mpg.quantile(1 / 3), 2)
    q2 = round(data.mpg.quantile(2 / 3), 2)
    data['category'] = ''
    data.category[data.mpg <= q1] = 'low'
    data.category[(data.mpg > q1) & (data.mpg <= q2)] = 'medium'
    data.category[data.mpg > q2] = 'high'
    return q1, q2, data


if __name__ == "__main__":
    """
    get the result of problem1 and print
    """
    data = read_data()
    q1, q2, data = set_category(data)
    print("The threshold is {} and {}.".format(q1, q2))
    print("low : mpg <= {}, sample size = {}.".format(q1, sum(data.category == 'low')))
    print("medium : {} < mpg <= {}, sample size = {}.".format(q1, q2, sum(data.category == 'medium')))
    print("high : mpg > {}, sample size = {}.".format(q2, sum(data.category == 'high')))

