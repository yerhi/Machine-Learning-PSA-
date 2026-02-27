# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:04:18 2020

@author: u0136350
"""


from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

cols_list = ['id','Group','BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli']
csv = pd.read_csv('C:/Users/u0136350/Documents/KU_Brain/PSA/Models/SignTasks1.csv',na_values = ':', sep=';', usecols = cols_list, engine='python')

data = pd.DataFrame(csv, columns = ['Group','BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli'])

features = pd.DataFrame(csv, columns = ['BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli'])



x = features.loc[:, ['BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli']]
y = data.Group


x = features.to_numpy()  #transforms dataframe into an array/matrix
y = y.to_numpy()

print(x.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

svc = svm.SVC()
svc.fit(X_train , y_train)
yhat_svc = svc.predict(X_test)
print(metrics.accuracy_score(yhat_svc , y_test))


