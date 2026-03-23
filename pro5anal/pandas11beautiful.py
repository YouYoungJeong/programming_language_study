"""  css 셀렉터를 이용 select(), select_one()"""
import requests
from bs4 import BeautifulSoup
html_page='''
<html>
<body>
<div id="hello">
    <a href="https://www.naver.com">naver</a>
    <span>
        <a href="https://www.daum.net">daum</a><br>
    </span>
    <ul class="world">
        <li>안녕</li>
        <li>반가워</li>
    </ul>
</div>
<div id="hi" class="good">
    두번째 div
</div>
</body>
</html>
'''

soup = BeautifulSoup(html_page, 'html.parser')
# aa = soup.select_one('div')
# aa = soup.select_one('div#hello')     # id가 hello인 div tag
# aa = soup.select_one('div.good')      # class가 good인 div tag
aa = soup.select_one('div#hello > a')   # 직계자식만 잡힘.
print("aa : ", aa ,' ', aa.string) 
print()
# bb = soup.select("div")
bb = soup.select('div#hello > ul.world')    # 자식
# bb = soup.select('div#hello ul.world')      # 자손
bb = soup.select('div#hello ul.world > li')
print("bb : ", bb ) 
for i in bb:
    print(i, ' : ', i.text)

print("-"*15," 위키백과 사이트에서 이순신으로 검색된 자료 읽기 ","-"*15)
url = "https://ko.wikipedia.org/wiki/이순신"
headers = {"User-Agent" : "Mozilla/5.0"}
wiki = requests.get(url=url, headers=headers)
# print(wiki.text[:100])
soup = BeautifulSoup(wiki.text, 'html.parser')
result = soup.select("p#mwHw")
# print(result) # 엘리먼트를 다 가져옴

# text만 뽑아오기
for s in result:
    for sup in s.find_all("sup"): # .<sup abou.. 이후는 필요 없어
        sup.decompose() # tag삭제
    print(s.get_text(strip=True))
    #추후 형태소 분리에서 품사별로 잘라서 문자열 판독할 때 많이 사용.
print()
print("=============== mw-content-text 문단 전체 읽기 ================")
# mw-content-text 문단 전체div id
result2 = soup.select("#mw-content-text p") # 직계아닌 자손 p
# print(result2) # 엘리먼트를 다 가져옴

# text만 뽑아오기
for s in result2:
    for sup in s.find_all("sup"): # .<sup abou.. 이후는 필요 없어
        sup.decompose() # tag삭제
    print(s.get_text(strip=True))
# 원하는 작업결과를 얻기위해 이런 작업을 많이 해야함.

print()
print("-"*15," 교촌치킨 사이트에서 메뉴 ,값 읽어서 DF만들기 ","-"*15)
# .asp 확장자가 .net으로 작업했구나.
import pandas as pd
url = "https://kyochon.com/menu/chicken.asp"
headers = {"User-Agent" : "Mozilla/5.0"}

response = requests.get(url, headers=headers)
# print(response.text) # 모든 사이트가 다 읽히지 않음. 셀레니움 사용(손작업이 많이 들어감)
soup2 = BeautifulSoup(response.text ,'html.parser')
# 메뉴명 얻기
# names = soup2.select("dl.txt > dt")
# print(names)
names = [tag.text.strip() for tag in soup2.select("dl.txt > dt")]
# print(names) # 이름이 잘 들어옴
prices = [int(tag.text.strip().replace(',','')) for tag in soup2.select("p.money strong")] # p tag의 자손
# replace(',','') - 콤마 없애서 출력
# print(prices) # 가격도 int로 잘 들어옴
## 네트워크가 불안할때 df로 만들어서 저장
df = pd.DataFrame({'상품명':names, "가격":prices})
print(df.head(3))
print(f'가격의 평균 : {df["가격"].mean():.2f}')
print(f'가격의 표준편차 : {df["가격"].std():.2f}')

"""
변동계수(CV, Coefficient of Variation)
데이터의 표준편차를 평균으로 나눈 비율로, 
측정 단위가 다르거나 평균 차이가 큰 데이터셋의 상대적 변동성을 비교하는 데 사용.
공식은 CV = (표준편차 / 평균) × 100%이며, 낮을수록 데이터의 정밀도가 높음을 의미.
100 : 완전히 흩어져 있음. 0 : 변동 X
"""
cv = df['가격'].std() / df['가격'].mean() * 100
print(f"가격 변동계수(CV) : {cv:.2f}%")
# 해석 : 가격 변동계수(CV) : 28.31%이므로 평균대비 적당히 퍼져있는 편이다.