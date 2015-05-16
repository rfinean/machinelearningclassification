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
X = dataArray[:,2:32].astype(float)
y = dataArray[:, 1]

print X.shape

yFreq = scipy.stats.itemfreq(y)
print yFreq

# Visualise frequencies of Malignant vs Benign

# plt.bar(left=0, height=int(yFreq[0][1]), color="red")
# plt.bar(left=1, height=int(yFreq[1][1]))
# plt.xlabel("diagnosis")
# plt.ylabel("frequency")
# plt.legend(['B', 'M'], loc='best')
# plt.show()

le = preprocessing.LabelEncoder()
le.fit(y)
yTransformed = le.transform(y)

# Correlation heat map

# correlationMatrix = numpy.corrcoef(X, rowvar=0)
# fig, ax = plt.subplots()
# heatmap = ax.pcolor(correlationMatrix, cmap=plt.cm.coolwarm_r)
# plt.show()

# Single scatter chart

plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
plt.xlabel("radius")
plt.ylabel("texture")
plt.show()


def scatter_plot(X, y):
    vs = X.shape[1]
    plt.figure(figsize=(2*vs, 2*vs))
    for i in range(vs):
        for j in range(vs):
            plt.subplot(vs, vs, i+1+j*vs)
            if i == j:
                plt.hist(X[:, i][y == "M"], alpha=0.4, color='m',
                    bins=numpy.linspace(min(X[:, i]), max(X[:, i]), 30))
                plt.hist(X[:, i][y == "B"], alpha=0.4, color='b',
                    bins=numpy.linspace(min(X[:, i]), max(X[:, i]), 30))
                plt.xlabel(i)
            else:
                plt.gca().scatter(X[:, i], X[:, j], c=y, alpha=0.4)
                plt.xlabel(i)
                plt.ylabel(j)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
    plt.tight_layout()
    plt.show()

scatter_plot(X[:, :5], y)
