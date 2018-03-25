from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

import numpy as np

#importing Digits dataset from sklearn.datasets
#dataset = datasets.load_digits()
dataset = datasets.fetch_mldata("MNIST Original")


features = np.array(dataset.data, "int16")        #Arrays containing all features of dataset's digits
labels = np.array(dataset.target, "int")          #Labels of dataset [0..9]

#Calculating HOG "Histogram of Oriented Gaussians"

list_hg_fod = []
for feature in features:
    #Reshaping image features to cells in each block equal to one and each individual cell is of size 14×14.
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14),
             cells_per_block=(1, 1), visualise=False)
    list_hg_fod.append(fd)

hog_features = np.array(list_hg_fod, "float64")

#Creating our classier
classifier = LinearSVC()
classifier.fit(hog_features, labels)

#Saving classifier in an external file

joblib.dump(classifier, "digits_classifier.pkl", compress=3)