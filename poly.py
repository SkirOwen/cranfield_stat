import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
plt.style.use('seaborn')


# Get the data
df = pd.read_csv("train_selected.csv")
X_train = df[["cycle", "s1", "s2", "s3", "s4"]]
y_train = df["ttf"]

X_test = pd.read_csv("test_selected.csv")
id_list = X_test["id"].tolist()
X_test = X_test.drop(columns=["id"])
y_test = pd.read_csv("PM_truth.txt", names=["ttf"])

# Defining the degree of the polynomial fit, 3 seemed to be a good value, for computation and results, also after
# messing around with finding_fit.py and scaling y differently to get a linear relationship
poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

polyreg = LinearRegression()
polyreg.fit(X_train_poly, y_train)

y_test_predict = polyreg.predict(X_test_poly)
y_train_predict = polyreg.predict(X_train_poly)

rmse = metrics.mean_squared_error(y_test, y_test_predict, squared=False)
mse = metrics.mean_squared_error(y_test, y_test_predict, squared=True)
r_2 = metrics.r2_score(y_test, y_test_predict)
sigma = metrics.explained_variance_score(y_test, y_test_predict)

title = " rmse".ljust(len(str(rmse)) - 6) + "|" + " mse".ljust(len(str(mse)) - 5) + "|" + " r2".ljust(len(str(r_2)) - 5)\
        + "|" + " Explained Variance"
print(title)
print("".ljust(len(title), "="))
print("{:10.6f}".format(rmse), "|", "{:10.6f}".format(mse), "|", "{:10.6f}".format(r_2), " |", "{:10.6f}".format(sigma))

print("&", "{:10.4f}".format(r_2),
      "&", "{:10.4f}".format(rmse),
      "&", "{:10.4f}".format(mse),
      "&", "{:10.4f}".format(sigma),
      "\\\\"
      )

# Graph of predicted values of the test data against the real test data

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

# Graph of the residual, I didn't used them in the report as I was confident of their relevance for every model tested

plt.scatter(y_train_predict, y_train_predict - y_train, s=30, label='Training data')
plt.scatter(y_test_predict, y_test_predict - y_test['ttf'].tolist(), marker='s', s=30, label='Test data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-50, xmax=400, color='red', lw=2)
plt.show()
