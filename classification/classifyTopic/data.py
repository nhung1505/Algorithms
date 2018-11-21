from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from bs4 import BeautifulSoup
import numpy as np
from xml.dom import minidom
import string
import _pickle as pickle


a = minidom.parse('traindatatopic.xml')
document = a.getElementsByTagName('document')
train_label = []
train_data = []
for doc in document:
    train_label.append(doc.getElementsByTagName("label")[0].childNodes[0].data)
    train_data.append(doc.getElementsByTagName("content")[0].childNodes[0].data.replace('\n', '.'))
train_data = train_data[:3000]
train_label = train_label[:3000]
train_data = [''.join(char for char in i if char not in string.punctuation) for i in train_data]    
stop_words = open('vietnamese-stopwords-dash.txt', 'r').read()
vectors = TfidfVectorizer(stop_words=stop_words.split('\n'), ngram_range=(1,3))


# training_corpus = np.vstack((X, y)).T
# train_data = []
# train_label = []
# A = []
# B = []
# C = []
# D = []
# E = []
# F = []
# for i in training_corpus:
#     if i[1] == 'foreign language':
#         A.append(i)
#     elif i[1] == 'advertisement':
#         B.append(i)
#     elif i[1] == 'other topics':
#         C.append(i)
#     elif i[1] == 'purchase':
#         D.append(i)
#     elif i[1] == 'recruit':
#         E.append(i)
#     else:
#         F.append(i)
    
# R = 12*A + B*2 + C*6 + D*8 + E*11 +F
# for row in R:
#     train_data.append(row[0])
#     train_label.append(row[1])

train_vec = vectors.fit_transform(train_data)

model = svm.SVC(kernel='linear')
model.fit(train_vec, train_label)

filename = 'model'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(vectors, open("tfidf_model", "wb"))
