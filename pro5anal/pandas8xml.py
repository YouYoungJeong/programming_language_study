import xml.etree.cElementTree as etrr
# 가져오는 데이터가 XML인경우 etrr - 돔을 이용해서 가져옴.
# 그외 뷰티풀숩
xmlfile = etrr.parse("my.xml")
print(xmlfile, type(xmlfile)) # <class 'xml.etree.ElementTree.ElementTree'>
root = xmlfile.getroot()
print(root.tag) # <Element 'items' at 0x000002C609DEB6A0> html의 body, head개념
print(root[0].tag) # root 요소의 0번째 요소명(노드명) 얻기
print(root[0][0].tag)
#----  더 쉬운 방법 있다
print()
myname = root.find('item').find('name').text
mytel = root.find('item').find('tel').text
print(myname + " " + mytel)
print()

print("-"*15," 기상청 제공 XML 자료 읽기(온도, 지역) ","-"*15)
import requests
# 웹에서 데이터를 가져오기위해 사용
# 단순한 문장열로 들어옴
# Xml을 처리하기위해 Etree사용

url = 'https://www.kma.go.kr/XML/weather/sfc_web_map.xml' # 기상청 실 데이터
headers = {"User-Agent":"Mozilla/5.0"} # 브라우저 정보를 알려주는것 쓰는건 고정 (Mozilla는 크롬브라우저 엔진)

res = requests.get(url, headers=headers)
res.raise_for_status() # 요청실패시 err발생 표시
print(res.text, type(res.text)) 
# <class 'str'> XML형태의 문자열. 
# 값을 뽑기가 어려워서 하지만 규칙은 있으니까 Etree를 사용해서 뽑아.

root= etrr.fromstring(res.text)
print(root) # <Element '{current}current' at 0x0000027B0FCC8130> 

# {current}current  namespace제거가 필요
for elem in root.iter():
    if "}" in elem.tag:
        elem.tag = elem.tag.split("}",1)[1]     #"}"를 기준으로 잘라 tag명만 남김. 
# {current}weather  -> weather

weather = root.find('weather')
year = weather.get('year') # '년도' 속성값을 얻는다.
month = weather.get('month')
day = weather.get('day')
hour = weather.get('hour')
print(f'{year}년 {month}월 {day}일 {hour}시 현재 예보')

# 각 지역(local tag) 순회
for local in weather.findall('local'):
    name = local.text.strip() # tag안에 text
    ta = local.get('ta') # local요소(엘리먼트)에 있는 ta속성(에트리뷰트)
    print(f"{name} 지역 온도는 {ta}")