"""
웹(동아일보) 에서 특정 단어 관련 문서들 검색 후 명사 만 추출.
    (가져오기 쉬운 사이트 이용하기 위해 동아일보 선택)
크롤링으로 데이터 얻어와서 워드 클라우드 그리기.
웹 크롤링할때 전부 같은 형태가 아니기 때문에 페이지소스보기도 참고해야함
WordCloud 시 필요한 설치 라이브러리
pip install pygame
pip install simplejson
pip install pytagcloud
"""
from bs4 import BeautifulSoup
from urllib.parse import quote
import urllib.request
from konlpy.tag import Okt
from collections import Counter     # 단어 수를 카운팅하는 라이브러리.
import pytagcloud
import matplotlib.pyplot as plt
import koreanize_matplotlib         # 한글처리
import matplotlib.image as mpimg
import webbrowser

# keyword = input('검색어 : ')
# # 인코딩 파싱.
# print(quote(keyword))

keyword = '춘분'

print("-"*40,"1) 검색페이지 잡기","-"*40)
target_url = "https://www.donga.com/news/search?query=" + quote(keyword)
# 1) 검색페이지 잡기 - urllib.request사용 일반 request랑 살짝 다름.
# source 코드 가져오기 - string
source_code = urllib.request.urlopen(target_url)
# print(source_code)
# soup 객체 변환
soup = BeautifulSoup(source_code,'lxml', from_encoding='utf-8')
# print(soup)
print()

print("-"*40,"2) 검색페이지 에서 링크 추출후 내용 추출(크롤링) ","-"*40)
# 2) 검색페이지 에서 링크 추출하기

msg = "" # 링크 담을  메세지 객체 생성.
# h4엘리먼트의 class가 tit
for title in soup.find_all("h4", class_="tit"):

    #h4엘리먼트의 class가 tit 안에 a테그 추출
    title_link = title.find('a')
    # print(title_link) # 확인.
    
    # a태그 안 href = url잡아오기
    article_url = title_link['href']
    # print(article_url)

        # 추출한 링크를 각각 다시 들어가서 '크롤링' 진행.
    try:
        source_article = urllib.request.urlopen(article_url)
        soup2 = BeautifulSoup(source_article, 'lxml', from_encoding='utf-8')
        # print(soup2)

        # 내용 읽을 변수 생성후 저장.
        contents = soup2.select('div.article_txt')
        # print(contents)

        # string만 꺼내기, 정규표현식도 사용하는게 좋은 명사분리할거라 괜찮.
        for imsi in contents:
            item = str(imsi.find_all(string=True))
            msg += item

    except Exception as err:
        pass

    # 마지막 문자열 확인.
    # print(msg)

    # 형태소 분석 후 명사 추출
    okt= Okt()
    nouns = okt.nouns(msg)

    # 한글자만 제외 후 리스트 저장
    result = []
    for imsi in nouns:
        if len(imsi) > 1 :
            result.append(imsi)

print("-"*40,"3) 형태소 분석 후 명사 추출, 카운딩 ","-"*40)
# print(result[:20])

# 카운팅하기 Counter모듈 사용하기
count = Counter(result)
# print(count)

# 상위 50개 추출 - wordCloud에 사용하기 위해
tag = count.most_common(50)
# print(tag)

print("-"*40,"4) 워드클라우드 작성 ","-"*40)
taglist = pytagcloud.make_tags(tag, maxsize=100) 
# [{'color': (176, 127, 139), 'size': 123, 'tag': '기운'},....
# print(taglist)

pytagcloud.create_tag_image(tags=taglist,
                            output='word.png', 
                            size=(1000, 600), 
                            background=(100,100,100),
                            fontname='Korean',
                            rectangular=False)
# 이미지 띄우기
img = mpimg.imread('word.png')
plt.imshow(img)
plt.show()

# 브라우저로 띄우기
# webbrowser.open('word.png')