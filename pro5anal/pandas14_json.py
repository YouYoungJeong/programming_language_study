"""
json 문서 처리 xml에 비해 경량 , 배열 개념만 있으면 처리 가능.
BeautifulSoup 은 json은 지원하지 X, 마크업데이터만 다룸
"""
import json
# dict 타입.
dict = {
    'name'  : 'tom',
    'age'   : 25,
    'score' : ['90','80','88']
}
print(dict, type(dict))
print("-"*15," json 인코딩 : dict -> str ","-"*15)
# json 인코딩 : dict -> str
str_val = json.dumps(dict)
# str_val = json.dumps(dict, indent=4) # 들여쓰기
print(str_val, type(str_val)) # <class 'str'> 외부로 넘기려면 str로 넘겨야함.
# print(str_val['name'])    # Err : str에는 이런 명령어가 없다
print(str_val[0:20])        # 문자열 관련 함수만 사용 가능
print()

print("-"*15," json 디코딩 : str -> dict ","-"*15)
json_val = json.loads(str_val)
print(json_val, type(json_val))
print(json_val['name'])     # dict관련 명령 사용 가능
for k in json_val.keys():
    print(k)

for v in json_val.values():
    print(v)
print()

import urllib.request as req
print("-"*15," 서울시 제공 도서관 정보 json 샘플 자료5개 읽기 ","-"*15)
url = "http://openapi.seoul.go.kr:8088/sample/json/SeoulLibraryTimeInfo/1/5/"
plainText = req.urlopen(url).read().decode()
print(plainText,type(plainText)) # <class 'str'>

# 디코딩 하기
jsonData = json.loads(plainText)
print(jsonData, type(jsonData)) # <class 'dict'>
print()

# dict 호출 하기
print('dict 호출 하기')
print(jsonData["SeoulLibraryTimeInfo"]["row"][0]["LBRRY_NAME"]) # BIBLIOTECA
print()
# dict의get()함수 사용
print("dict의 get()함수 사용 - dict 명령어 사용")
libData = jsonData.get("SeoulLibraryTimeInfo").get("row")
# print(libData)
name = libData[0].get('LBRRY_NAME')
print(name)
print()


datas = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    addr = ele.get('ADRES')
    print(name)
    print(tel)
    print(addr)
    datas.append([name, tel, addr])
import pandas as pd
df = pd.DataFrame(datas, columns=['도서관명','전화','주소'])
print(df)
