""" BeautifulSoup 객체 find 메소드 사용하기 """
from bs4 import BeautifulSoup

# 웹에서 가져왔다고 가정
html_page = """
            <html><body>
            <h1>제목 태그</h1>
            <p>웹 문서 연습</p>
            <p>원하는 자료 확인</p>
            </body></html>
            """
print(type(html_page)) #<class 'str'>
soup = BeautifulSoup(html_page, 'html.parser')
# str -> BeautifulSoup클래스 객체로 변환 -> 그래야 BeautifulSoup이 제공하는 메소드 사용가능
print(type(soup)) # <class 'bs4.BeautifulSoup'>
print()
# 돔 구조 형태로 불러옴
"""
string과 text의 차이
    h1.string
        태그 안에 문자열이 딱 하나만 있을 때 그 문자열을 가져옴.
        내부에 다른 태그가 섞여 있으면 None 이 나올 수 있음.
    h1.text
        태그 안의 모든 텍스트를 전부 합쳐서 가져옴.
        내부에 하위 태그가 있어도 텍스트를 추출함
"""
h1 = soup.html.body.h1
print("h1 : ", h1.string) # string = text
print("h1 : ", h1.text)

p1 = soup.html.body.p # 최초의 p
print("p1 : ", p1.string)

# DOM을 이용한 자료 접근. 현실적으로 문제가 있음.
p2 = p1.next_sibling.next_sibling  # (sibling)형제 태그 출력
print("p2 : ", p2.string)
print()

print("-"*15," BeautifulSoup find() method 사용하기 ","-"*15)
""" 
find(tag명, attrs(attribute:속성값), recursive(자손검색 T/F) , string)   
find() - 처음만난 하나만 읽음.   
find_all() - 

"""
html_page2 = """
            <html><body>
            <h1 id='title'>제목 태그</h1>
            <p>웹 문서 연습</p>
            <p id='my' class='our'>원하는 자료 확인</p>
            </body></html>
            """
soup2 = BeautifulSoup(html_page2, 'html.parser')

""" 속성값 선택하기 """
print(soup2.p, ' ', soup2.p.string)
print(soup2.find('p').string) # find(['p','h1']) # 이렇게하면 복수개가 잡힘.
# id
print("--id--")
print(soup2.find('p', id='my').string)
print(soup2.find(id='title').string) # 속성값만 적어도 출력됨 id는 유니크이기 때문.
print(soup2.find(id='my').string)
print("--class--")
# class 는 키워드이기 때문에 그냥 쓰면 err -> 'class_'로 적어야 한다
print(soup2.find(class_='our').string)
print("--attrs--")
# attrs를 사용하는 경우는 attrs{key : value}값으로 넣음.
print(soup2.find(attrs={'class':'our'}).string)
print(soup2.find(attrs={'id':'my'}).string)
print()

print("-"*15," BeautifulSoup find_all() method 사용하기 ","-"*15)
html_page3 = """
            <html><body>
            <h1 id='title'>제목 태그</h1>
            <p>웹 문서 연습</p>
            <p id='my' class='our'>원하는 자료 확인</p>
            <div>
                <a href='https://www.naver.com'>naver</a><br>
                <a href='https://www.daum.net'>daum</a>
            </div>
            </body></html>
            """
soup3 = BeautifulSoup(html_page3, 'html.parser')
print(soup3.find_all(['a'])) # list안에 a tag모두 선택됨.
print(soup3.find_all(['a','p']))
print()
links = soup3.find_all('a')
# print(links)
for i in links:
    href = i.attrs['href']
    text = i.text
    print(href, " ", text)

print("-"*15," 정규 표현식 사용 ","-"*15)
import re
links2 = soup3.find_all(href=re.compile(r'^https'))
for k in links2:
    print(k.attrs['href'])

print()

print("-"*15," bugs 사이트 음악 순위 읽기 ","-"*15)
import requests
url = "https://music.bugs.co.kr/chart"
response = requests.get(url)
# print(response.text)
bsoup = BeautifulSoup(response.text ,'html.parser')
musics = bsoup.find_all("td", class_="check")
for idx, music in enumerate(musics):
    print(f"{idx + 1}위 {music.input['title']}")
    