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
y = []
X = []
for doc in document:
    y.append(doc.getElementsByTagName("label")[0].childNodes[0].data)
    X.append(doc.getElementsByTagName("content")[0].childNodes[0].data.replace('\n', '.'))
train_data = X[:2500]
train_label = y[:2500]
train_data = [''.join(char for char in i if char not in string.punctuation) for i in train_data]    
stop_words = open('vietnamese-stopwords-dash.txt', 'r').read()
vectors = TfidfVectorizer(stop_words=stop_words.split('\n'), ngram_range=(1,3))


training_corpus = np.vstack((train_data, train_label)).T
train_data = []
train_label = []
A = []
B = []
C = []
D = []
E = []
F = []
for i in training_corpus:
    if i[1] == 'foreign language':
        A.append(i)
    elif i[1] == 'advertisement':
        B.append(i)
    elif i[1] == 'other topics':
        C.append(i)
    elif i[1] == 'purchase':
        D.append(i)
    elif i[1] == 'recruit':
        E.append(i)
    else:
        F.append(i)
    
R = 9*A + B + C*3 + D*5 + E*6 +F
for row in R:
    train_data.append(row[0])
    train_label.append(row[1])
print(len(train_data))


train_vec = vectors.fit_transform(train_data)

model = svm.SVC(kernel='linear')
model.fit(train_vec, train_label)

filename = 'model'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(vectors, open("tfidf_model", "wb"))
