from bs4 import BeautifulSoup
import requests
file = 'alonhadat.txt'
with open(file) as f:
    content = f.readlines()
content = [x.strip() for x in content] 
for url in content:
    response = requests.get(url)
    data = response.text
    soup = BeautifulSoup(data, 'lxml')
    tags = soup.findAll('a')
    char = 'tel:'
    for tag in tags:
        x = tag.get('href')
        if x.find(char) >= 0:
            a = tag.get('href').replace(char, '')
            with open('phone.txt', 'a') as fileUrl:
                fileUrl.writelines('%s\n' % a)
