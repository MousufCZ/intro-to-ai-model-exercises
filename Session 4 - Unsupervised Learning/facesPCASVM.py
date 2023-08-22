
"""
Ex4

@author: jacob
"""

import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

#import that faces dataset
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

#show some faces
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])
plt.show()
plt.close()

#build a PCA model with 150 components
pca = PCA(n_components=150, whiten=True, random_state=42) 
pca.fit(faces['data'])

#plot explained varianace against components
plt.figure
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()
plt.close()

#split the dataset
Xtrain, Xtest, ytrain, ytest = train_test_split(faces['data'], faces['target'],
                                                random_state=42)

#transform the data
projection = pca.transform(Xtrain)
projectionTest = pca.transform(Xtest)

print(Xtrain.shape)
print(projection.shape)

#set up an SVC (parameters are important here) and fit the data, labelled this time
svc = SVC(kernel='rbf', class_weight='balanced', gamma=0.001, C=5)
svc.fit(projection, ytrain)

#make predictions
y_pred = svc.predict(projectionTest)

#calculate accuracy score and print
print('Accuracy: %.2f' % accuracy_score(ytest, y_pred))

#use the classification report to get a more details summary of successes
from sklearn.metrics import classification_report
print(classification_report(ytest, y_pred, target_names=faces.target_names))

#show the PCA representations of the faces, and colour
#code labels for success or failure.
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(pca.components_[i].reshape(62, 47), cmap='bone') 
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[y_pred[i]].split()[-1],
                   color='black' if y_pred[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);



