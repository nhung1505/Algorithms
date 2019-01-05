from pymongo import MongoClient
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests

client = MongoClient('localhost', 27017)
db = client.forumsdb
collection = db.alonhadat

linkUrls = []
urls = []
for x in collection.find():
    linkUrls.append(x['Url'])

for link in linkUrls:
    request = requests.get(link)
    a =request.status_code
    if request.status_code == 200:
        with open('alonhadat.txt', 'a') as fileUrl:
            fileUrl.writelines('%s\n' % link)


