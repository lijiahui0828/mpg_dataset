import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')

from basicfunction import read_data
from problem1 import set_category

data = read_data()
_, _, data = set_category(data)

# draw the scatterplot matrix
sns.set(style="ticks", color_codes=True, font_scale=1.5)
g = sns.pairplot(data.iloc[:, 1:], hue="category", diag_kind="hist", markers = ['o','D','s'])
g.fig.set_size_inches(20,20)
plt.show()