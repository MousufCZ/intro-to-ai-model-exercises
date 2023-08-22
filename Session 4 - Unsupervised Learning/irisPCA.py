
"""
Ex4

@author: jacob
"""

#import libraries used
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture     
from sklearn.cluster import KMeans

#method to plot confusion matrices
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#load the dataset from sklearn
iris = datasets.load_iris()

#find the two principal componants and project onto these
pca = PCA(n_components=2)
pca.fit(iris.data)
projected = pca.fit_transform(iris.data)

print(projected.shape)

#now plot this PCA projection
plt.scatter(projected[:, 0], projected[:, 1],
            c=iris.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('rainbow', 3))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show;

#added for plenary session
from sklearn.svm import SVC

svmMod = SVC(kernel='linear')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(projected, iris['target'], test_size=0.25, random_state=7)
svmMod.fit(X_train, y_train)

y_pred = svmMod.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
#end added for plenary session

#set up a kMeans model, with three cluster, and fit the data to it
kMeans = KMeans(n_clusters=3)  
kMeans.fit(iris['data'])                   

#find clusters
kClusters = kMeans.fit_predict(iris['data'])

#now map the data onto the clustering
y_kmeans = kMeans.predict(iris['data'])

print(y_kmeans)
print(iris['target'])

#map the discovered clusters to the labels, the calculate accuracy
kLabels = np.zeros_like(kClusters)
for i in range(3):
    mask = (kClusters == i)
    kLabels[mask] = mode(iris['target'][mask])[0]

print('Accuracy: %.2f' % accuracy_score(iris['target'], kLabels))

#find the confusion matrix, normalise and print
cmk = confusion_matrix(iris['target'], kLabels)
np.set_printoptions(precision=2)
cmk_normalized = cmk.astype('float') / cmk.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cmk_normalized)

#confusion matric as a figure
plt.figure()
plot_confusion_matrix(cmk_normalized, iris.target_names, title='Normalized confusion matrix')
plt.show()


#set up a GMM model, and fit the data to it
model = GaussianMixture(n_components=3)  
model.fit(iris['data'])                   

#find clusters
clusters = model.fit_predict(iris['data'])

#now map the data onto the clustering
y_gmm = model.predict(iris['data'])

print(y_gmm)
print(iris['target'])

#map the discovered clusters to the labels, the calculate accuracy
labels = np.zeros_like(clusters)
for i in range(3):
    mask = (clusters == i)
    labels[mask] = mode(iris['target'][mask])[0]

print('Accuracy: %.2f' % accuracy_score(iris['target'], labels))

#find the confusion matrix, normalise and print
cm = confusion_matrix(iris['target'], labels)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

#confusion matrix as a figure
plt.figure()
plot_confusion_matrix(cm_normalized, iris.target_names, title='Normalized confusion matrix')
plt.show()