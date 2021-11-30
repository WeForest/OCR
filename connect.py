#from urllib.request import urlopen
#import urllib

#url='http://3.35.222.251:5000/abuse?data=sefef'

#data = urlopen(url)
#print(data.read(500).decode('utf-8'))

'''
import requests
myurl = 'http://34.125.102.123:5000/fileUpload'
files = {'image': open('png.png', 'rb')}
getdata = requests.post(myurl, files=files)
print(getdata.text)  
'''


a = ['삼성', '주니어', '소프트웨어', '창작대회']
b = ['김보석', '주니어', '삼성', '소프트웨어', '보이즈', '창작대회']
if a in b :
    print('he')