from matplotlib import pyplot as plt
import numpy as np


file = 'phone.txt'
with open(file) as f:
    phones = f.readlines()
phones = [x.strip() for x in phones]

list = {}
for phone in phones:
    if phone not in list :
        list[phone] = 1
    else :
        list[phone] += 1
sale = 0
nsale = 0
y =[]
X = []
for i in list:
    phone = i 
    x = list[i]
    if x >1:
        sale += 1
    else:
        nsale += 1
        print(i)
    X.append(x)
    y.append(phone)
print(sale)
print(nsale)

plt.plot(X)
plt.ylabel('số bài:')
plt.show()

