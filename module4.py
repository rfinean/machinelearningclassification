import csv
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#Imports for Module 4
from sklearn import neighbors
import knnplots


#Code common to all modeles from module 3 onwards
##NB. The X and yTransformed variables come from the preprocessing in the previous module.
fileName = "wdbc.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen)
dataList = list(csvData)
dataArray = numpy.array(dataList)
X = dataArray[:, 2:32].astype(float)
y = dataArray[:, 1]
le = preprocessing.LabelEncoder()
le.fit(y)
yTransformed = le.transform(y)


nbrs = neighbors.NearestNeighbors(n_neighbors=10, algorithm="ball_tree").fit(X)
distances, indices = nbrs.kneighbors(X)

# print indices[:5]
# print distances[:5]


def predict(k, X, y, Xunknown):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights="distance")     # not weights="uniform"
    knn = knn.fit(X, y)
    return knn.predict(Xunknown)

# Perfect predictions there test data = training data
predictedK3 = predict(3, X, yTransformed, X)
predictedK15 = predict(15, X, yTransformed, X)
nonAgreement = predictedK3[predictedK3 != predictedK15]
print "No of K3 vs K15 discrepencies =", len(nonAgreement)

falseK3 = predictedK3[predictedK3 != yTransformed]
falseK15 = predictedK15[predictedK15 != yTransformed]
print "No of K3 errors =", len(falseK3)
print "No of K15 errors =", len(falseK15)

# Split the data 75% into training set and 25% into test data
XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

predictedWU = predict(3, XTrain, yTrain, XTest)

print metrics.classification_report(yTest, predictedWU)
print metrics.accuracy_score(yTest, predictedWU)

knnplots.plotaccuracy(XTrain, yTrain, XTest, yTest, 300)

knnplots.decisionplot(XTrain, yTrain, n_neighbors=3, weights="uniform")

knnplots.decisionplot(XTrain, yTrain, n_neighbors=15, weights="uniform")
