import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def get_regression_metrics(model, actual, predicted):
    """Calculate main regression metrics.

    Args:
        model (str): The model name identifier
        actual (series): Contains the test label values
        predicted (series): Contains the predicted values

    Returns:
        dataframe: The combined metrics in single dataframe


    """
    regr_metrics = {
        "Root Mean Squared Error": metrics.mean_squared_error(actual, predicted) ** 0.5,
        "Mean Absolute Error": metrics.mean_absolute_error(actual, predicted),
        "R^2": metrics.r2_score(actual, predicted),
        "Explained Variance": metrics.explained_variance_score(actual, predicted)
    }

    # return reg_metrics
    df_regr_metrics = pd.DataFrame.from_dict(regr_metrics, orient="index")
    df_regr_metrics.columns = [model]
    return df_regr_metrics


# Get the data
df = pd.read_csv("train_selected.csv")
X_train = df[["cycle", "s1", "s2", "s3", "s4"]]
y_train = df["ttf"]

X_test = pd.read_csv("test_selected.csv")
id_list = X_test["id"].tolist()
X_test = X_test.drop(columns=["id"])
y_test = pd.read_csv("PM_truth.txt", names=["ttf"])

poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

polyreg = LinearRegression()
polyreg.fit(X_train_poly, y_train)

y_test_predict = polyreg.predict(X_test_poly)
y_train_predict = polyreg.predict(X_train_poly)

# print("R^2 training: %s, R^2 test: %s" % (
#       (metrics.r2_score(y_train, y_train_predict)),
#       (metrics.r2_score(y_test, y_test_predict))))
#
# polyreg_metrics = get_regression_metrics("Polynomial Regression", y_test, y_test_predict)
# print(polyreg_metrics)


rmse = metrics.mean_squared_error(y_test, y_test_predict, squared=False)
mse = metrics.mean_squared_error(y_test, y_test_predict, squared=True)
r_2 = metrics.r2_score(y_test, y_test_predict)
sigma = metrics.explained_variance_score(y_test, y_test_predict)

print("Scores\n==============================")
print("rmse = ", rmse)
print("mse = ", mse)
print("r2 = ", r_2)
print("explained variance = ", sigma)

print(" &", r_2, "&", rmse, "&", mse, "&", sigma, "\\\\")


fig, axs = plt.subplots(2, 3)

axs[0, 0].scatter(np.arange(1, 101), y_test_predict, alpha=0.75, s=50)
axs[0, 0].scatter(np.arange(1, 101), y_test["ttf"].tolist(), c="#DD8452", alpha=0.75, s=50)

axs[0, 1].scatter(X_test["s1"].tolist(), y_test_predict, alpha=0.75, s=50)
axs[0, 1].scatter(X_test["s1"].tolist(), y_test["ttf"].tolist(), c="#DD8452", alpha=0.75, s=50)

axs[0, 2].scatter(X_test["s2"].tolist(), y_test_predict, alpha=0.75, s=50)
axs[0, 2].scatter(X_test["s2"].tolist(), y_test["ttf"].tolist(), c="#DD8452", alpha=0.75, s=50)

axs[1, 1].scatter(X_test["s3"].tolist(), y_test_predict, alpha=0.75, s=50)
axs[1, 1].scatter(X_test["s3"].tolist(), y_test["ttf"].tolist(), c="#DD8452", alpha=0.75, s=50)

axs[1, 2].scatter(X_test["s4"].tolist(), y_test_predict, alpha=0.75, s=50)
axs[1, 2].scatter(X_test["s4"].tolist(), y_test["ttf"].tolist(), c="#DD8452", alpha=0.75, s=50)

axs[1, 0].scatter(X_test["cycle"].tolist(), y_test_predict, alpha=0.75, s=50)
axs[1, 0].scatter(X_test["cycle"].tolist(), y_test["ttf"].tolist(), c="#DD8452", alpha=0.75, s=50)

plt.setp(axs[0, 0], xlabel='id')
plt.setp(axs[0, 1], xlabel='s1')
plt.setp(axs[0, 2], xlabel='s2')
plt.setp(axs[1, 0], xlabel='cycle')
plt.setp(axs[1, 1], xlabel='s3')
plt.setp(axs[1, 2], xlabel='s4')

plt.setp(axs[:, 0], ylabel='ttf')
plt.show()

plt.scatter(y_train_predict, y_train_predict - np.array(y_test["ttf"]), c="blue", marker="o", label="Training data")
plt.scatter(y_test_predict, y_test_predict - np.array(y_test["ttf"]), c="lightgreen", marker="s", label="Test data")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=-50, xmax=400, color="red", lw=2)
plt.show()

