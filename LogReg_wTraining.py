# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:30:46 2020

@author: u0136350

Geraldine Rodriguez Nieto - PSA
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns



cols_list = ['id','Group','BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt']
csv = pd.read_csv('C:/Users/u0136350/Documents/KU_Brain/PSA/Models/SignTasks3.csv',na_values = ':', sep=';', usecols = cols_list, engine='python')

data = pd.DataFrame(csv, columns = ['Group','SIMKAPstresstolerance','DTomission','SMKMeanAngleDev','BTTSubAnDiff113','DTrt'])



feature_cols = ['SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','BTTSubAnDiff113'] #including rt ddecreases accuracy


y = data.Group

X= data[feature_cols]
#X = features.to_numpy()  #transforms dataframe into an array/matrix
y = y.to_numpy()

print(X.shape)
print(y.shape)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)

logreg = LogisticRegression()

model = logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
print(model.coef_) #prints betas
print(model.intercept_) #prints constant/intercept from formula


