import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn')

from sklearn import linear_model, neighbors, tree, svm, ensemble, metrics, model_selection, isotonic
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler, LabelBinarizer

import pydotplus
from IPython.display import Image

lb = LabelBinarizer()

# Get the data
df = pd.read_csv("train_selected.csv")
X = df[["cycle", "s1", "s2", "s3", "s4"]]


test = pd.read_csv("test_selected.csv")
real_value = pd.read_csv("PM_truth.txt", names=["ttf"])

# y = df["ttf"]
y = df["label_bnc"]
real_value = pd.read_csv("PM_truth.txt", names=["label_bnc"])
real_value["label_bnc"] = real_value["label_bnc"].apply(lambda x: 0 if x > 30 else 1)

show_tree = False

# y = np.power(y, 2/np.pi)

# model to change

regr = ensemble.RandomForestClassifier(max_depth=8, min_samples_leaf=17, random_state=17)
# regr = GaussianNB()
# regr = linear_model.LogisticRegression(max_iter=250, random_state=17, n_jobs=-1)
# regr = linear_model.Lasso(random_state=17)
# regr = linear_model.ElasticNet(random_state=17)
# regr = linear_model.BayesianRidge()
# regr = linear_model.TweedieRegressor(power=0)
# regr = linear_model.TweedieRegressor(power=1)

# regr = tree.DecisionTreeRegressor(max_depth=8, min_samples_leaf=17, random_state=17)
# regr = ensemble.RandomForestRegressor(n_estimators=500, oob_score=True, random_state=17, n_jobs=-1)

# regr = tree.DecisionTreeClassifier(max_depth=8, min_samples_leaf=17, random_state=17)
# regr = ensemble.ExtraTreesClassifier(max_depth=8, min_samples_leaf=17, random_state=17)
# regr = ensemble.ExtraTreesRegressor(max_depth=8, min_samples_leaf=17, random_state=17)
# regr = ensemble.ExtraTreesRegressor(max_depth=8, min_samples_leaf=17, random_state=17)


# regr = svm.LinearSVC(random_state=17)
# regr = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=8, min_samples_leaf=17, random_state=17), n_estimators=500, random_state=17)
# regr = AdaBoostRegressor(ensemble.ExtraTreesRegressor(max_depth=8, min_samples_leaf=17, random_state=17), n_estimators=500, random_state=17)

# regr = neighbors.KNeighborsClassifier(n_jobs=-1, n_neighbors=2)
# regr = neighbors.KNeighborsRegressor(n_jobs=-1)
# regr = svm.SVC(kernel="rbf", probability=True, random_state=17)
# regr = ensemble.GradientBoostingRegressor(random_state=17)
#
# regr = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=17)


# Voting
# clf1 = linear_model.LinearRegression(n_jobs=-1)
# clf2 = ensemble.RandomForestRegressor(max_depth=8, min_samples_leaf=17, random_state=17, n_jobs=-1)
# clf3 = ensemble.GradientBoostingRegressor(random_state=17)
# # clf3 = svm.SVC(kernel="rbf", probability=True, random_state=17)
#
# regr = ensemble.VotingRegressor(
#     estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], n_jobs=-1
# )

regr.fit(X, y)
y_train_predict = regr.predict(X)
expected_values = []

for i in range(len(test)):
    guessed_value = regr.predict([[*test.iloc[i, 1:]]])
    expected_values.append(guessed_value)

expected_values = np.array(expected_values)

id_list = test['id'].tolist()
try:
    real_value_list = real_value['ttf'].tolist()
except KeyError:
    real_value_list = real_value['label_bnc'].tolist()

# real_value_list = np.power(real_value_list, 2/np.pi)

# Scores:
#
print('Accuracy', metrics.accuracy_score(real_value_list, expected_values), "\n",
      'macro F1', metrics.f1_score(real_value_list, expected_values, average='macro'), "\n",
      'micro F1', metrics.f1_score(real_value_list, expected_values, average='micro'), "\n",
      'macro Precision', metrics.precision_score(real_value_list, expected_values, average='macro'), "\n",
      'micro Precision', metrics.precision_score(real_value_list, expected_values, average='micro'), "\n",
      'macro Recall', metrics.recall_score(real_value_list, expected_values, average='macro'), "\n",
      'micro Recall', metrics.recall_score(real_value_list, expected_values, average='micro'), "\n",
      'macro ROC AUC', metrics.roc_auc_score(real_value_list, expected_values, average='macro'), "\n",
      'micro ROC AUC', metrics.roc_auc_score(real_value_list, expected_values, average='micro'))

print(' &', "{:10.4f}".format(metrics.accuracy_score(real_value_list, expected_values)), '&',
      "{:10.4f}".format(metrics.f1_score(real_value_list, expected_values, average='macro')),
      '&', "{:10.4f}".format(metrics.f1_score(real_value_list, expected_values, average='micro')), '&',
      "{:10.4f}".format(metrics.precision_score(real_value_list, expected_values, average='macro')),
      '&', "{:10.4f}".format(metrics.precision_score(real_value_list, expected_values, average='micro')), '&',
      "{:10.4f}".format(metrics.recall_score(real_value_list, expected_values, average='macro')),
      '&', "{:10.4f}".format(metrics.recall_score(real_value_list, expected_values, average='micro')), '&',
      "{:10.4f}".format(metrics.roc_auc_score(real_value_list, expected_values, average='macro')),
      '&', "{:10.4f}".format(metrics.roc_auc_score(real_value_list, expected_values, average='micro')))

rmse = metrics.mean_squared_error(real_value_list, expected_values, squared=False)
mse = metrics.mean_squared_error(real_value_list, expected_values, squared=True)
r_2 = metrics.r2_score(real_value_list, expected_values)
sigma = metrics.explained_variance_score(real_value_list, expected_values)

print("Scores\n==============================")
print("rmse = ", rmse)
print("mse = ", mse)
print("r2 = ", r_2)
print("explained variance = ", sigma)

print(" &", r_2, "&", rmse, "&", mse, "&", sigma, "\\\\")
# print("AUC-ROC = ", metrics.roc_auc_score(real_value_list, expected_values))

# print("depth = ", regr.get_depth())
# print("leaves = ", regr.get_n_leaves())

fig, axs = plt.subplots(2, 3)

axs[0, 0].scatter(np.arange(1, 101), expected_values, alpha=0.75, s=50)
axs[0, 0].scatter(np.arange(1, 101), real_value_list, c="#DD8452", alpha=0.75, s=50)

axs[0, 1].scatter(test["s1"].tolist(), expected_values, alpha=0.75, s=50)
axs[0, 1].scatter(test["s1"].tolist(), real_value_list, c="#DD8452", alpha=0.75, s=50)

axs[0, 2].scatter(test["s2"].tolist(), expected_values, alpha=0.75, s=50)
axs[0, 2].scatter(test["s2"].tolist(), real_value_list, c="#DD8452", alpha=0.75, s=50)

axs[1, 1].scatter(test["s3"].tolist(), expected_values, alpha=0.75, s=50)
axs[1, 1].scatter(test["s3"].tolist(), real_value_list, c="#DD8452", alpha=0.75, s=50)

axs[1, 2].scatter(test["s4"].tolist(), expected_values, alpha=0.75, s=50)
axs[1, 2].scatter(test["s4"].tolist(), real_value_list, c="#DD8452", alpha=0.75, s=50)

axs[1, 0].scatter(test["cycle"].tolist(), expected_values, alpha=0.75, s=50)
axs[1, 0].scatter(test["cycle"].tolist(), real_value_list, c="#DD8452", alpha=0.75, s=50)

plt.setp(axs[0, 0], xlabel='id')
plt.setp(axs[0, 1], xlabel='s1')
plt.setp(axs[0, 2], xlabel='s2')
plt.setp(axs[1, 0], xlabel='cycle')
plt.setp(axs[1, 1], xlabel='s3')
plt.setp(axs[1, 2], xlabel='s4')

plt.setp(axs[:, 0], ylabel='ttf')
plt.show()

plt.scatter(y_train_predict, y_train_predict - y, s=30, label='Training data')
plt.scatter(expected_values, expected_values - real_value, marker='s', s=30, label='Test data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-50, xmax=400, color='red', lw=2)
plt.show()

if show_tree:
    print(tree.export_graphviz(regr, None))

    # Create Dot Data
    dot_data = tree.export_graphviz(regr,
                                    out_file=None)  # Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
    # Create Graph from DOT data
    graph = pydotplus.graph_from_dot_data(dot_data)

    # Show graph
    Image(graph.create_png())

    # Create PNG
    graph.write_png("tree.png")
