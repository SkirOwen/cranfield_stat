import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns


df = pd.read_csv("train_selected.csv")
df = df.drop(columns=["id", "ttf", "label_bnc"])

sns.lineplot(data=df, palette="tab10", linewidth=2.5)
plt.show()

