import pandas as pd
import numpy as np
from pandas import plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train = pd.read_csv('data/train.csv/train.csv')
test = pd.read_csv('data/test_v2.csv/test_v2.csv')

print(train.shape)
print(train.isnull().sum())

#plotting.scatter_matrix(train.iloc[:, 1:], figsize = (8, 8), c = list(train.iloc[:, 0]), alpha = 0.5)
#plt.show()

df = train.drop(['time', 'state', 'car_value', 'risk_factor', 'C_previous', 'duration_previous'], axis = 1)
train_X = df.drop('record_type', axis = 1)
train_y = df.record_type

(train_X, test_X, train_y, test_y) = train_test_split(train_X, train_y, test_size = 0.3, random_state = 666)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(train_X, train_y)
pred = clf.predict(test_X)

print(pred)