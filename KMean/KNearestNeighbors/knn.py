# Importing libraries
import pandas as pd
import numpy as np
import math
import operator
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Importing data
trainingSet = []
testSet = []
with open("/home/nguyen_nhung/vs/Algorithms/KMean/KNearestNeighbors/iris.csv", 'r') as file:
    content = csv.reader(file)
    data = list(content)
    for x in range(len(data) - 1):
        if random.random() < 0.5:
            trainingSet.append(data[x])
        else:
            testSet.append(data[x])
        
# print(trainingSet[:5])
# print(testSet[:5])

# Defining a function which calculates euclidean distance between two data points
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += pow(data1[x] - data2[x], 2)
    return np.sqrt(distance)



# Defining our KNN model
def knn(trainingSet, testInstance, k):
    distances = []
    sort = []
    length = len(testInstance) - 1

     # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))

     # sorting them on the basic of distance
    sorted_d = sorted(distances, key=operator.itemgetter(1))
    neighbors = []

     # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    
    classVotes = {}

    # calculating the most freg class in the neighbors
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)

# creating a dummy testset
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

# setting number of neighbors = 1
k = 2
# running Knn model
result, neigh = knn(trainingSet, test, k)

# y_pred = clf.predict(testSet)
# predicted class
print(result)
# nearest neighbor
print(neigh)




       
   

