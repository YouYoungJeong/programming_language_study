'''
웹 문서를 읽어 형태소 분석 : 위키백과에서 단어 검색 결과
단어 출력 횟수를 DataFrame형태로 출력.
'''
import requests
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd
from urllib import parse # 한글 인코딩 해주는 lib
import warnings # 경고 문자 끄기
warnings.filterwarnings(action='ignore')
okt = Okt()

#url을 줄때 %EC%9D%B4%EC%88%9C%EC%8B%A0 인코딩이 끝난 상태(바이너리 형태)의 한글데이터
# url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"

# 인코딩을 하지 않는 url
para0 = '이순신'
url0 = "https://ko.wikipedia.org/wiki/" + para0
print(url0) # https://ko.wikipedia.org/wiki/이순신 인코딩이 필요함.

# 인코딩 끝난 URL 생성 - parse모듈 사용
para = parse.quote('이순신')
url = "https://ko.wikipedia.org/wiki/" + para
print(url)

headers = {'User-Agent':'Mozilla/5.0'}

response = requests.get(url, headers=headers)

if response.status_code == 200: # 통신상태 양호 + 내가 요청한 정보 정확히 클라이언트에게 넘어감.
    page = response.text
    # print((page), type(page)) # <class 'str'>
    soup = BeautifulSoup(page, "lxml")

    # 형태소 분석으로 명사 추출해 기억하기 위한 list 생성
    wordlist = []

    # id="mw-content-text" 자손인 ptag를 가져올거야
    for item in soup.select("#mw-content-text p"):

        # item에 string이 있으면 list에 저장
        if item.string != None:
            wordlist += okt.nouns(item.string)
    print(wordlist)
    print('단어 수 : ', len(wordlist))
    print('중복 제거 후 단어 수: ', len(set(wordlist)))
    print()
    
    # 단어의 발생 횟수 dict에 저장. 누적카운트
    word_dict = {} 
    for i in wordlist:
        if i in word_dict:
            word_dict[i] += 1
        else:
            word_dict[i] = 1
    print(word_dict)
    print()

    print("-"*40,"list - Series로 출력","-"*40)
    seri_list = pd.Series(wordlist)
    print(seri_list[:3])
    print(seri_list.value_counts()[:5])
    print()

    print("-"*40,"dict - Series로 출력","-"*40)
    seri_dict = pd.Series(word_dict)
    print(seri_dict[:3])
    print(seri_dict.value_counts()[:5])
    print()

    print("-"*40,"list - DataFrame 출력","-"*40)
    df1 = pd.DataFrame(wordlist, columns=['단어'])
    print(df1.head(3))
    print()

    # konlpy 로 작업하는 시간이 오래걸리기 때문에 작업이 끝나면 저장하는것이 좋다.
    print("-"*40,"dict - DataFrame 출력","-"*40)
    df2 = pd.DataFrame([word_dict.keys(), word_dict.values()])
    df2 = df2.T
    df2.columns = ['단어','빈도수']
    print(df2.head())
    print("-"*40,"csv파일로 저장","-"*40)
    df2.to_csv('nlp_morph2.csv', index=False)
    df3 = pd.read_csv('nlp_morph2.csv')
    print(df3.head())
