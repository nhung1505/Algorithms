from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from bs4 import BeautifulSoup
import numpy as np
from xml.dom import minidom
import string


a = minidom.parse('traindatatopic.xml')
document = a.getElementsByTagName('document')
y = []
X = []
for doc in document:
    y.append(doc.getElementsByTagName("label")[0].childNodes[0].data)
    X.append(doc.getElementsByTagName("content")[0].childNodes[0].data.replace('\n', '.'))

X = [''.join(char for char in i if char not in string.punctuation) for i in X]    
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

train_vec = vectors.fit_transform(X)
model = svm.SVC(kernel='linear')
model.fit(train_vec, y)



train_data, test_data, train_label, test_label = train_test_split(X, y, test_size = 0.3 , random_state = 42)
# print(len(train_data))
# print(len(test_data))
test_vec = vectors.transform(test_data)
prediction = model.predict(test_vec)
print (classification_report(test_label, prediction))
test_data_exam = ['Một trong những sự kiện HOT nhất song hành cùng buổi OFFLINE FC 17/11 sắp tới, Cuộc thi hát Online ĐHFS VOICE đã chính thức ra mắt chào đón tất cả các giọng ca vàng FC Love Đinh Hương ở khắp mọi nơi !!! Các cá nhân/nhóm tham gia cover lại những ca khúc của Đinh Hương và quay video clip, đăng tải lên Youtube và gửi về theo link đăng ký dưới đây: http://bit.ly/16FS0oU Và đặc biệt, Phần Giải thưởng rất hấp dẫn: Giải Nhất (Album SOUL, áo FC, huy hiệu FC, và 1 BỮA ĂN THÂN MẬT VỚI ĐINH HƯƠNG), ngoài ra còn có Giải Khán Giả Bình Chọn nữa nheeee!!! Xem Thể Lệ Cuộc Thi:']
test_vec_exam = vectors.transform(test_data_exam)
pre = model.predict(test_vec_exam)
print(pre)
