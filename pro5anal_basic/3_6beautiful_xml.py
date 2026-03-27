"""
BeautifulSoup 모듈로 XML 문서 처리
"""
from bs4 import BeautifulSoup

with open('my.xml', mode='r', encoding="utf-8") as f:
    xmlfile = f.read()
    print(xmlfile, type(xmlfile)) #  <class 'str'>

# type가 str이니 객체로 만들기
soup = BeautifulSoup(xmlfile, 'lxml')
print(type(soup)) # <class 'bs4.BeautifulSoup'>
itemTag = soup.find_all('item')
print(itemTag[0])
print()
nameTag =soup.find_all('name')
print(nameTag[0]['id'])
print('--------------------------')
for i in itemTag:
    nameTag = i.find_all('name')
    for j in nameTag:
        print("id : " + j['id'] + " | name : " + j.string)
        tel = i.find('tel')
        print("tel : " + tel.string)
    
    for j in i.find_all('exam'):
        print("kor : "+ j['kor'] + ", eng :" + j['eng'])
    print()

print("-"*15," 서울시 제공 도서관 정보 XML 샘플 자료5개 읽기 ","-"*15)
import urllib.request as req # 교육용으로 많이 사용
import pandas as pd
url = "http://openapi.seoul.go.kr:8088/sample/xml/SeoulLibraryTimeInfo/1/5/"
plainText = req.urlopen(url=url).read().decode()
# print(plainText)
xmlObj = BeautifulSoup(plainText, 'xml')

# row만 필요함
libData = xmlObj.select('row')
# print(libData)

# 도서관 정보를 기억할 리스트
rows = []
for data in libData:
    name = data.find("LBRRY_NAME").text
    addr = data.find("ADRES").text
    print(f"도서관명 : {name}\n주소 : {addr}")
    print()
    rows.append({"도서관명":name, "주소":addr})
# print(rows)
df = pd.DataFrame(rows)
print(df)
print('건수 : ' , len(df))
