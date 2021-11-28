#from urllib.request import urlopen
#import urllib

#url='http://3.35.222.251:5000/abuse?data=sefef'

#data = urlopen(url)
#print(data.read(500).decode('utf-8'))


import requests
myurl = 'http://3.35.222.251:5000/fileUpload'
files = {'image': open('png.png', 'rb')}
getdata = requests.post(myurl, files=files)
print(getdata.text)  