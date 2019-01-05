from pymongo import MongoClient
import numpy as np

client = MongoClient('localhost', 27017)
db = client.forumsdb
collection = db.Articles
print(collection)

file = open('domains.txt', 'r')
charbds = [ 'nhadat', 'batdongsan', 'bannha']
arrFile = []
domainRemoves = []

for line in file:
    arrFile.append(line)

Domains = [s.replace('\n', '') for s in arrFile]


for  domain in Domains:
    for charbd in charbds:
        if(domain.find(charbd) >=0 ):
            domainRemoves.append(domain)



for domainRemove in domainRemoves:
    for domain in Domains:
        if domain == domainRemove:
            Domains.remove(domainRemove)




def removeDomain(a):
    collection.remove({"Domain" : a})

for domain in Domains:
    print(domain)
    removeDomain(domain)
