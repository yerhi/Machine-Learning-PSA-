# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:39:52 2020

@author: u0136350
"""





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.model_selection import train_test_split


cols_list = ['id','Group','BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli']
csv = pd.read_csv('C:/Users/u0136350/Documents/KU_Brain/PSA/Models/SignTasks1.csv',na_values = ':', sep=';', usecols = cols_list, engine='python')

data = pd.DataFrame(csv, columns = ['Group','BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli'])

features = pd.DataFrame(csv, columns = ['BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli'])


n_neighbors = 15


# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset


x = features.loc[:, ['BTTSubAnDiff113','SIMKAPstresstolerance','SMKMeanAngleDev','DTomission','DTrt','DTNumStimuli']]
y = data.Group


x = features.to_numpy()  #transforms dataframe into an array/matrix
y = y.to_numpy()

print(x.shape)
print(y.shape)


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 



#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report
from sklearn import metrics
metrics.confusion_matrix(y_pred, y_test)
print(classification_report(y_test, y_pred))




#print("confusion matrix: \n" + str(metrics.confusion_matrix(y_pred , y))

#print 'Predicted value: ' + str(yhat [-1]),
#’, real target: ’ + str(y[ -1])














#
#h = .02  # step size in the mesh

# Create color maps
#cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
#cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

#for weights in ['uniform', 'distance']:
 #   # we create an instance of Neighbours Classifier and fit the data.
  #  clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
   # clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
       #                  np.arange(y_min, y_max, h))
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    #Z = Z.reshape(xx.shape)
    #plt.figure()
    #plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
     #           edgecolor='k', s=20)
    #plt.xlim(xx.min(), xx.max())
   # plt.ylim(yy.min(), yy.max())
   # plt.title("3-Class classification (k = %i, weights = '%s')"
       #       % (n_neighbors, weights))

#plt.show()
