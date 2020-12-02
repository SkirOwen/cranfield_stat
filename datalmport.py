import numpy as np
import pandas as pd
import os

"""
Ignore this file, this was an attempt to translate the matlab code into python, but while doing so I realise
that python already has the tools built-in to do it.
"""


def dataImport(dirName, isPlot, EID=1):
    # Training dataset

    train = pd.read_csv((os.path.join(dirName, "train_selected.csv")))
    # Train = csvread('train_selected.csv', 1, 0);

    id = train[:, 0]
    cycle = train[:, 1]

    s1 = train[:, 2]
    s2 = train[:, 3]
    s3 = train[:, 4]
    s4 = train[:, 5]

    ttf = train[:, 6]
    label_bnc = train[:, 7]

    test = pd.read_csv((os.path.join(dirName, "test_selected.csv")))

    return trainingData, testData

