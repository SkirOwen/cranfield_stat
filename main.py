import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import linear_model


df = pd.read_csv("train_selected.csv")

test = pd.read_csv("test_selected.csv")
real_value = pd.read_csv("PM_truth.txt", names=["ttf"])

X = df[["cycle", "s1", "s2", "s3", "s4"]]
y = df["ttf"]

# model to change
regr = linear_model.LinearRegression()
regr.fit(X, y)

expected_values = []

for i in range(len(test)):
	guessed_value = regr.predict([[*test.iloc[i, 1:]]])
	expected_values.append(guessed_value)

id_list = test['id'].tolist()
real_value_list = real_value['ttf'].tolist()

plt.figure()
plt.grid(b=True, which='both', axis='x', markevery=1)
plt.scatter(id_list, expected_values, c='b')
plt.scatter(id_list, real_value_list, c='r')
plt.show()


# TODO: error

# regr.fit(X, y)


# df = df.drop(columns=["id", "ttf", "label_bnc"])

# sns.pairplot(df, hue="label_bnc")
# sns.jointplot("s1", "ttf", data=df, kind='reg')

# with sns.axes_style('white'):
#     sns.jointplot("s1", "ttf", df, kind='hex')

# g = sns.PairGrid(df, vars=['cycle', 's1', 's2', 's3', 's4', 'ttf'],
#                  hue='label_bnc')
# g.map(plt.scatter, alpha=0.8)
# g.add_legend()

# plt.show()

