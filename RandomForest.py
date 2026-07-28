# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:13:11 2020

@author: u0136350

Geraldine Rodriguez Nieto - PSA
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics


import matplotlib.pyplot as plt
import seaborn as sns

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

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#feature importance scores:
feature_imp = pd.Series(clf.feature_importances_,index=['BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli']).sort_values(ascending=False)


%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()





