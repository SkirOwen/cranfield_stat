import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv("train_selected.csv")

test = pd.read_csv("test_selected.csv")
real_value = pd.read_csv("PM_truth.txt", names=["ttf"])

X = df[["cycle", "s1", "s2", "s3", "s4"]]
y = df["ttf"]

# 2D
# plt.figure()
# plt.scatter(X.iloc[:, 1], np.power(y, 2/np.pi))
# plt.show()

# 3D
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X.iloc[:, 1], X.iloc[:, 2], y)
plt.show()
