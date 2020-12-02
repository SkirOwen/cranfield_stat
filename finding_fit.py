import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

"""
The purpose of this file is just to mess around to get an idea of the data I'll be dealing with.

Uncomment what you want
"""

df = pd.read_csv("train_selected.csv")
df = df.drop(columns=["id"])


test = pd.read_csv("test_selected.csv")
real_value = pd.read_csv("PM_truth.txt", names=["ttf"])

X = df[["cycle", "s1", "s2", "s3", "s4"]]
y = df["ttf"]

# 2D
# plt.figure()
# plt.scatter(X.iloc[:, 1], np.power(y, 2/np.pi))
# plt.show()

# 3D
# fig = plt.figure()
# ax = Axes3D(fig)
#
# ax.scatter(X.iloc[:, 1], X.iloc[:, 2], y)
# plt.show()

sns.pairplot(df, hue="label_bnc")

# g = sns.PairGrid(df, hue='label_bnc')
# g.map_diag(sns.kdeplot)
# g.map_offdiag(sns.scatterplot)
# g.add_legend()
plt.show()
