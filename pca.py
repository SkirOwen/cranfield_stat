import numpy as np
np.random.set_state(17)

import pandas as pd
import prince

df = pd.read_csv("train_selected.csv")
df = df.drop(columns=["id"])


test = pd.read_csv("test_selected.csv")
real_value = pd.read_csv("PM_truth.txt", names=["ttf"])

X = df[["cycle", "s1", "s2", "s3", "s4"]]
y = df["ttf"]

pca = prince.PCA(n_components=3,
                 n_iter=17,
                 rescale_with_mean=False,
                 rescale_with_std=False,
                 copy=True, random_state=17
                 )

pca = pca.fit(X)
pca.transform(X)
print(pca.explained_inertia_)



