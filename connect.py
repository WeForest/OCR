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
names = ''
data = ['전도업', '2021', '대전광역시', '제56외', '전국기능경기대회', 'kormhwwwwwwwworar', '선수', '모바일로보틱스']
for i in data:
    if len(i) <= 3:
        names = names + i + ","
print(names)

