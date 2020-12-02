import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

plt.style.use('seaborn')


# Variable for the user to change
model = "logistic"
print_for_latex = False
show_tree = False
table_print = True

REGRESSION_MODEL = [
    "linear",
    "lasso",
    "elastic",
    "bayesian",
    "elasticnet",
    "decision tree regressor", "dtr",
    "tweedie regressor 0", "normal distribution",
    "tweedie regressor 1", "poisson distribution",
    "extra trees regressor", "etr",
    "random forest regressor", "rfr",
    "adaboost extra trees", "boost et",
    "k neighbours", "k neighbor",
    "gradient boosting regressor", "gbr",
    "voting"
]

CLASSIFICATION_MODEL = [
    "logistic",
    "gaussian",
    "decision tree classifier", "dtc",
    "extra tree classifier", "etc",
    "random forest classifier", "rfc",
    "k neighbour classifier", "k neighbor classifier",
    "linear svc",
    "svc"
]

# Get the data
df = pd.read_csv("train_selected.csv")
X = df[["cycle", "s1", "s2", "s3", "s4"]]

test = pd.read_csv("test_selected.csv")
real_value = pd.read_csv("PM_truth.txt", names=["ttf"])

model = model.lower()

if model in REGRESSION_MODEL:
    mode = "regression"
elif model in CLASSIFICATION_MODEL:
    mode = "classification"
else:
    raise Exception("Please use one of these models:", *REGRESSION_MODEL, "or", *CLASSIFICATION_MODEL)

# Get the correct y according to the model
if mode == "regression":
    y = df["ttf"]
elif mode == "classification":
    # Labeling the PM_truth.txt as the training data
    y = df["label_bnc"]
    real_value = pd.read_csv("PM_truth.txt", names=["label_bnc"])
    real_value["label_bnc"] = real_value["label_bnc"].apply(lambda x: 0 if x > 30 else 1)


def model_selector(model):
    # Ye good ol ugly if-elif switch to choose the model
    if model == "linear":
        regr = linear_model.LinearRegression(n_jobs=-1)

    elif model == "lasso":
        regr = linear_model.Lasso(random_state=17)

    elif model == "elasticnet" or model == "elastic":
        regr = linear_model.ElasticNet(random_state=17)

    elif model == "bayesian":
        regr = linear_model.BayesianRidge()

    elif model == "decision tree regressor" or model == "dtr":
        regr = tree.DecisionTreeRegressor(max_depth=8, min_samples_leaf=17, random_state=17)

    elif model == "tweedie regressor 0" or model == "normal distribution":
        regr = linear_model.TweedieRegressor(power=0)

    elif model == "tweedie regressor 1" or model == "poisson distribution":
        regr = linear_model.TweedieRegressor(power=1)

    elif model == "extra trees regressor" or model == "etr":
        regr = ensemble.ExtraTreesRegressor(max_depth=8, min_samples_leaf=17, random_state=17)

    elif model == "random forest regressor" or model == "rfr":
        regr = ensemble.RandomForestRegressor(n_estimators=500, oob_score=True, random_state=17, n_jobs=-1)

    elif model == "adaboost extra trees" or model == "boost et":
        regr = AdaBoostRegressor(ensemble.ExtraTreesRegressor(max_depth=8, min_samples_leaf=17, random_state=17),
                                 n_estimators=500, random_state=17)

    elif model == "k neighbours" or model == "k neighbor":
        regr = neighbors.KNeighborsRegressor(n_jobs=-1)

    elif model == "gradient boosting regressor" or model == "gbr":
        regr = ensemble.GradientBoostingRegressor(random_state=17)

    elif model == "voting":
        clf1 = linear_model.LinearRegression(n_jobs=-1)
        clf2 = ensemble.RandomForestRegressor(max_depth=8, min_samples_leaf=17, random_state=17, n_jobs=-1)
        clf3 = ensemble.GradientBoostingRegressor(random_state=17)
        regr = ensemble.VotingRegressor(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], n_jobs=-1)

    elif model == "logistic":
        regr = linear_model.LogisticRegression(max_iter=250, random_state=17, n_jobs=-1)

    elif model == "gaussian":
        regr = GaussianNB()

    elif model == "decision tree classifier" or model == "dtc":
        regr = tree.DecisionTreeClassifier(max_depth=8, min_samples_leaf=17, random_state=17)

    elif model == "extra tree classifier" or model == "etc":
        regr = ensemble.ExtraTreesClassifier(max_depth=8, min_samples_leaf=17, random_state=17)

    elif model == "random forest classifier" or model == "rfc":
        regr = ensemble.RandomForestClassifier(max_depth=8, min_samples_leaf=17, random_state=17)

    elif model == "linear svc":
        regr = svm.LinearSVC(random_state=17)

    elif model == "k neighbour classifier" or model == "k neighbor classifier":
        regr = neighbors.KNeighborsClassifier(n_jobs=-1, n_neighbors=2)

    elif model == "svc":
        regr = svm.SVC(kernel="rbf", probability=True, random_state=17)

    return regr


regr = model_selector(model)

regr.fit(X, y)
y_train_predict = regr.predict(X)

expected_values = []
for i in range(len(test)):
    guessed_value = regr.predict([[*test.iloc[i, 1:]]])
    expected_values.append(guessed_value)
expected_values = np.array(expected_values)

# Convert dataframe in list to plot
id_list = test['id'].tolist()
try:
    real_value_list = real_value['ttf'].tolist()
except KeyError:
    real_value_list = real_value['label_bnc'].tolist()


if table_print:
    if mode == "classification":
        accuracy = metrics.accuracy_score(real_value_list, expected_values)
        macro_f1 = metrics.f1_score(real_value_list, expected_values, average='macro')
        micro_F1 = metrics.f1_score(real_value_list, expected_values, average='micro')
        macro_prec = metrics.precision_score(real_value_list, expected_values, average='macro')
        micro_prec = metrics.precision_score(real_value_list, expected_values, average='micro')
        macro_recall = metrics.recall_score(real_value_list, expected_values, average='macro')
        micro_recall = metrics.recall_score(real_value_list, expected_values, average='micro')
        macro_roc = metrics.roc_auc_score(real_value_list, expected_values, average='macro')
        micro_roc = metrics.roc_auc_score(real_value_list, expected_values, average='micro')

        title = " Accuracy".ljust(10) + "|" +\
                " macro F1".ljust(10) + "|" +\
                " micro F1".ljust(10) + "|" +\
                " macro Precision".ljust(17) + "|" +\
                " micro Precision ".ljust(17) + "|" +\
                " macro Recall".ljust(14) + "|" +\
                " micro Recall".ljust(14) + "|" +\
                " macro ROC AUC".ljust(15) + "|" +\
                " micro ROC AUC".ljust(15)
        print(title)
        print("".ljust(len(title), "="))
        print(" {:.6f}".format(accuracy), "|",
              "{:.6f}".format(macro_f1), "|",
              "{:.6f}".format(micro_F1), "|",
              "{:.6f}".format(macro_prec).ljust(15), "|",
              "{:.6f}".format(micro_prec).ljust(15), "|",
              "{:.6f}".format(macro_recall).ljust(12), "|",
              "{:.6f}".format(micro_recall).ljust(12), "|",
              "{:.6f}".format(macro_roc).ljust(13), "|",
              "{:.6f}".format(macro_roc)
              )

        if print_for_latex:
            print(' &', "{:10.6f}".format(accuracy), '&',
                  "{:10.6f}".format(macro_f1), '&',
                  "{:10.6f}".format(micro_F1), '&',
                  "{:10.6f}".format(macro_prec), '&',
                  "{:10.6f}".format(micro_prec), '&',
                  "{:10.6f}".format(macro_recall), '&',
                  "{:10.6f}".format(micro_recall), '&',
                  "{:10.6f}".format(macro_roc), '&',
                  "{:10.6f}".format(micro_roc)
                  )

    if mode == "regression":
        rmse = metrics.mean_squared_error(real_value_list, expected_values, squared=False)
        mse = metrics.mean_squared_error(real_value_list, expected_values, squared=True)
        r_2 = metrics.r2_score(real_value_list, expected_values)
        sigma = metrics.explained_variance_score(real_value_list, expected_values)

        title = " rmse".ljust(len(str(rmse)) - 6) + "|" + " mse".ljust(len(str(mse)) - 5) + "|" + " r2".ljust(
            len(str(r_2)) - 5) + "|" + " Explained Variance"
        print(title)
        print("".ljust(len(title), "="))
        print("{:10.6f}".format(rmse), "|", "{:10.6f}".format(mse), "|", "{:10.6f}".format(r_2), " |",
              "{:10.6f}".format(sigma))

        print("&", "{:10.6f}".format(r_2),
              "&", "{:10.6f}".format(rmse),
              "&", "{:10.6f}".format(mse),
              "&", "{:10.6f}".format(sigma),
              "\\\\"
              )

        if print_for_latex:
            print(" &", r_2, "&", rmse, "&", mse, "&", sigma, "\\\\")


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

# Graph of the residual, I didn't used them in the report as I was confident of their relevance for every model tested

plt.scatter(y_train_predict, y_train_predict - y, s=30, label='Training data')
plt.scatter(expected_values.flatten(), expected_values.flatten() - real_value_list, marker='s', s=30, label='Test data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-50, xmax=400, color='red', lw=2)
plt.show()

# To create a .png of the tree if model == tree
if show_tree:
    print("depth = ", regr.get_depth())
    print("leaves = ", regr.get_n_leaves())

    print(tree.export_graphviz(regr, None))

    # Create Dot Data
    dot_data = tree.export_graphviz(regr, out_file=None)
    # Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes
    # or leaf nodes

    # Create Graph from DOT data
    graph = pydotplus.graph_from_dot_data(dot_data)

    # Show graph
    Image(graph.create_png())

    # Create PNG
    graph.write_png("tree.png")
