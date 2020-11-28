import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import linear_model, metrics, tree, ensemble

import pydotplus
from IPython.display import Image


df = pd.read_csv("train_selected.csv")

test = pd.read_csv("test_selected.csv")
real_value = pd.read_csv("PM_truth.txt", names=["ttf"])

X = df[["cycle", "s1", "s2", "s3", "s4"]]
y = df["ttf"]
y = np.power(y, 2/np.pi)

# model to change
regr = linear_model.LinearRegression()
# regr = tree.DecisionTreeRegressor(random_state=42)
# regr = tree.DecisionTreeRegressor(random_state=0)
# regr = tree.DecisionTreeClassifier(random_state=42)
# regr = ensemble.ExtraTreesClassifier(random_state=42)
# regr = ensemble.ExtraTreesRegressor(random_state=42)
regr.fit(X, y)

expected_values = []

for i in range(len(test)):
	guessed_value = regr.predict([[*test.iloc[i, 1:]]])
	expected_values.append(guessed_value)

id_list = test['id'].tolist()
real_value_list = real_value['ttf'].tolist()

real_value_list = np.power(real_value_list, 2/np.pi)

rmse = metrics.mean_squared_error(real_value_list, expected_values, squared=False)
print("rmse = ", rmse)

print("score = ", regr.score(test.iloc[0:, 1:], real_value_list))
# print("depth = ", regr.get_depth())
# print("leaves = ", regr.get_n_leaves())

plt.figure()
plt.grid(b=True, which='both', axis='x', markevery=1)
plt.scatter(id_list, expected_values, c='b')
plt.scatter(id_list, real_value_list, c='r')
plt.show()

# print(tree.export_graphviz(regr, None))
#
# #Create Dot Data
# dot_data = tree.export_graphviz(regr, out_file=None) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
# #Create Graph from DOT data
# graph = pydotplus.graph_from_dot_data(dot_data)
#
# # Show graph
# Image(graph.create_png())


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
