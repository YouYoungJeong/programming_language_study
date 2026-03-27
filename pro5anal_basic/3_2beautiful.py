"""
웹에서 자료 읽기에서 사용하는 대표적인 2가지 모듈
    requests : 좀더 많이 사용.
    urllib.request 은 교육용
읽어오기만 하면 그냥 문자열이기 때문에 BeautifulSoup을 이용해 객체화 함.
    requests.get()으로 웹 데이터를 가져옴
    response.text를 BeautifulSoup()에 넣어서 객체화함
    그다음 find(), select() 같은 걸로 원하는 데이터 추출함
    requests가 불러온 HTML을 BeautifulSoup이 분석하는 구조

참고 사이트:    https://cafe.daum.net/flowlife/RUrO/42
                https://www.crummy.com/software/BeautifulSoup/bs4/doc/
    크롤러 솔루션 scrapy :  https://engkimbs.tistory.com/893
                            https://scrapy.org/
python에서 대부분의 web 작업에 Beautiful Soup을 사용한다.

BeautifulSoup 객체를 이용한 웹 문서 처리 

Parser : BeautifulSoup(markup, "html.parser") / BeautifulSoup(markup, "lxml")을 가장 많이 사용.
ex : soup = BeautifulSoup("<html>a web page</html>", 'html.parser') html을 객체화함
추세 : find_all() == findAll()
    find() 첫번째 태그
    find_next() 다음 태그
    find_all()  모든 태그

요즘 웹크롤링에 대한 보안 강화 때문에 Selenium이 생김.
"""
import requests
from bs4 import BeautifulSoup

baseurl = "https://www.naver.com"

# 웹 요청 보낼때 서버에게 정보를 알림. 
# 이런 정보를 안 주면 request에서 데이터를 못 읽을 수 있다.
headers = {"User-Agent" : "Mozilla/5.0"} 

source = requests.get(baseurl, headers=headers)
print(source, type(source)) # <Response [200]> <class 'requests.models.Response'>
# 200 : 내가 요청한 파일이 정상적으로 넘어옴
print(source.status_code) # 200
# print(source.text) # 네이버 메인페이지의 내용이 전부다 넘어옴.
print(type(source.text)) # <class 'str'> 그냥 단순한 문자열
# print(source.content) # 바이너리 데이터로 넘어옴 - 실질적으로 데이터가 넘어올때 이렇게 옴,
print(type(source.content)) # <class 'bytes'>
print()

print("-"*15," BeautifulSoup 객체로 생성 ","-"*15)
"""    BeautifulSoup 객체로 생성   """

# xml로 Parser
conv_data = BeautifulSoup(source.text, "lxml") 
# print(conv_data)
print(type(conv_data)) # <class 'bs4.BeautifulSoup'> 뷰티풀숲 클래스 타입이 됨.

# a tag 읽어오기
for atag in conv_data.find_all("a"):
    # atag 형태 : <a href:''>hyper text </a>
    href = atag.get('href')
    title = atag.get_text(strip=True) # 앞뒤 띄어쓰기 없애고 받기
    # title이 있을경우만 출력
    if title:
        print(href)
        print(title)
        print('-------------')