# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:39:52 2020

@author: u0136350

Geraldine Rodriguez Nieto - KU Leuven
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import metrics

cols_list = ['id','Group','BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli']
csv = pd.read_csv('C:/Users/u0136350/Documents/KU_Brain/PSA/Models/SignTasks1.csv',na_values = ':', sep=';', usecols = cols_list, engine='python')

data = pd.DataFrame(csv, columns = ['Group','BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli'])

features = pd.DataFrame(csv, columns = ['BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli'])


n_neighbors = 15

x = features.loc[:, ['BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli']]
y = data.Group

x = features.to_numpy() 
y = y.to_numpy()

print(x.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

metrics.confusion_matrix(y_pred, y_test)
print(classification_report(y_test, y_pred))

