import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the data
df = pd.read_csv("train_selected.csv")
X = df[["cycle", "s1", "s2", "s3", "s4"]]
y = df["ttf"]


a = df.loc[df["id"] == 17]

test = pd.read_csv("test_selected.csv")
real_value = pd.read_csv("PM_truth.txt", names=["ttf"])

# fig, axs = plt.subplots(nrows=4)
# g1 = sns.lineplot(x="cycle", y="s1", hue="label_bnc", data=a, ax=axs[0])
# g2 = sns.lineplot(x="cycle", y="s2", hue="label_bnc", data=a, ax=axs[1])
# g2.legend(loc=3)
# g3 = sns.lineplot(x="cycle", y="s3", hue="label_bnc", data=a, ax=axs[2])
# g4 = sns.lineplot(x="cycle", y="s4", hue="label_bnc", data=a, ax=axs[3])
# g4.legend(loc=3)
# plt.show()
# #]

# df_corre = df[["s1", "s2", "s3", "s4", "ttf"]].corr()
#
# sns.set(font_scale=1.0)
# fig = plt.figure(figsize=(10, 8))
# hm = sns.heatmap(df_corre, cbar=True, annot=True, square=True, fmt=".2f")
# plt.title("Correlation Heatmap")
# # plt.show()
#
sns.pairplot(df, hue="label_bnc", diag_kind="kde", markers=".", plot_kws={"alpha": 0.6, "s": 3, "edgecolor": None})
plt.show()
