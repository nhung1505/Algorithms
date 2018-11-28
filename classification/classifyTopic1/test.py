from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from bs4 import BeautifulSoup
import numpy as np
from xml.dom import minidom
import string
import _pickle as pickle


a = minidom.parse('traindatatopic.xml')
filename = 'model'
load_model = pickle.load(open(filename, 'rb'))
document = a.getElementsByTagName('document')
test_label = []
test_data = []
for doc in document:
    test_label.append(doc.getElementsByTagName("label")[0].childNodes[0].data)
    test_data.append(doc.getElementsByTagName("content")[0].childNodes[0].data.replace('\n', '.'))

test_data = [''.join(char for char in i if char not in string.punctuation) for i in test_data]    
test_data = test_data[2500:]
test_label = test_label[2500:]
vectors = pickle.load(open("tfidf_model", "rb"))
test_vec = vectors.transform(test_data)
prediction = load_model.predict(test_vec)
print (classification_report(test_label, prediction))
print("có nhập câu???")
t = input()
test_data_exam = []
while(t != 'no'):
    print("input")
    i = input()
    print(i)
    test_data_exam.append(i)
    print("input?")
    print(test_data_exam)
    t = input()

test_vec_exam = vectors.transform(test_data_exam)
pre = load_model.predict(test_vec_exam)
print(pre)