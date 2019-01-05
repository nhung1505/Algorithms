from pymongo import MongoClient
import numpy as np


client = MongoClient('localhost', 27017)
db = client.forumsdb
collection = db.Articles
# print(collection.find_one())
X = collection.distinct("Domain")
arr =[]
for data in X:
    arr.append(data)
with open('domains.txt', 'w') as filehandle:  
    for listitem in arr:
        filehandle.write('%s\n' % listitem)
