import csv
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import neighbors
import knnplots
from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


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
XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

knnK3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knnK15 = neighbors.KNeighborsClassifier(n_neighbors=15)
nbmodel = GaussianNB()

knn3scores = cross_validation.cross_val_score(knnK3, XTrain, yTrain, cv=5)
print knn3scores
print "Mean of scores KNN3 = ", knn3scores.mean()
print "S.D. of scores KNN3 = ", knn3scores.std()

knn15scores = cross_validation.cross_val_score(knnK15, XTrain, yTrain, cv=5)
print knn15scores
print "Mean of scores KNN15 = ", knn15scores.mean()
print "S.D. of scores KNN15 = ", knn15scores.std()

nbscores = cross_validation.cross_val_score(nbmodel, XTrain, yTrain, cv=5)
print nbscores
print "Mean of scores NB = ", nbscores.mean()
print "S.D. of scores NB = ", nbscores.std()


meansKNNK3 = []
sdsKNNK3 = []
meansKNNK15 = []
sdsKNNK15 = []
meansNB = []
sdsNB = []

ks = range(2, 21)

for k in ks:
    knn3scores = cross_validation.cross_val_score(knnK3, XTrain, yTrain, cv=k)
    knn15scores = cross_validation.cross_val_score(knnK15, XTrain, yTrain, cv=k)
    nbscores = cross_validation.cross_val_score(nbmodel, XTrain, yTrain, cv=k)
    meansKNNK3.append(knn3scores.mean())
    sdsKNNK3.append(knn15scores.std())
    meansKNNK15.append(knn15scores.mean())
    sdsKNNK15.append(knn15scores.std())
    meansNB.append(nbscores.mean())
    sdsNB.append(nbscores.std())

plt.plot(ks, meansKNNK3, label="KNN 3 accuracy", color="purple")
plt.plot(ks, meansKNNK15, label="KNN 15 accuracy", color="yellow")
plt.plot(ks, meansNB, label="NB accuracy", color="blue")
plt.legend(loc="best")
plt.title("Accuracy vs K")
plt.ylim(0.9, 1)
plt.show()

plt.plot(ks, sdsKNNK3, label="KNN 3 SD accuracy", color="purple")
plt.plot(ks, sdsKNNK15, label="KNN 15 SD accuracy", color="yellow")
plt.plot(ks, sdsNB, label="NB SD accuracy", color="blue")
plt.legend(loc="best")
plt.title("Standard Deviation vs K")
plt.ylim(0, 0.1)
plt.show()
