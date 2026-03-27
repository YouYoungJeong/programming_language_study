"""
https://finance.naver.com/sise/sise_market_sum.naver?&page=1~2페이지 읽어서
col = [종목명	현재가	전일비	등락률	액면가	시가총액	상장주식수	외국인비율	거래량	PER	ROE]
df에 저장 후 
with open(파일명, mode='W'....) -> CSV파일로 출력

---------------------------
CSV파일로 읽기 후 DataFrame에 저장
top 3 종목명. 시가총액 출력
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

"""---------------------------------------------------------------------"""
urls = [
    "https://finance.naver.com/sise/sise_market_sum.naver?&page=1",
    "https://finance.naver.com/sise/sise_market_sum.naver?&page=2"
]

headers = {"User-Agent": "Mozilla/5.0"}
file_name = "market_cap.csv"


with open(file_name, mode='w', encoding='utf-8') as f:
    f.write("종목명,시가총액\n")
    
    for url in urls:
        res = requests.get(url=url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')

        rows = soup.select("table.type_2 > tbody > tr")
        for row in rows:
            if not row.select_one("a.tltle"): continue

            name = row.select_one("a.tltle").get_text(strip=True)
            price = row.select(".number")[4].get_text(strip=True).replace(',', '')
            
            f.write(f"{name},{price}\n")

df = pd.read_csv(file_name)
df['시가총액'] = pd.to_numeric(df['시가총액'])
df.index = df.index + 1
print(df[['종목명', '시가총액']].head(5))
"""---------------------------------------------------------------------"""

url = "https://finance.naver.com/sise/sise_market_sum.naver?&page=1"
headers = {"User-Agent" : "Mozilla/5.0"} 

# col name 추출
col = soup.find_all(attrs={'scope':'col'})
col_name = []
for i in range(1,len(col)-1):
    col_name += [col[i].text]

df_list = []
for i in range(1,3):
    res = requests.get(f"https://finance.naver.com/sise/sise_market_sum.naver?&page={i}", headers)
    soup = BeautifulSoup(res.text, 'html.parser')


    # name 추출
    names = soup.find_all("a", attrs={'class' : "tltle"})
    df_names = []
    for name in names:
        df_names += ([name.text]) 


    # data값 추출
    data = soup.find_all(attrs={'class':'number'})
    print(data)
    data2 = [tag.text.strip().replace("\n\t\t\t\t",' ') for tag in data]
    # print(data2)
    data3 = []
    for a in range(0, 500, 10):
            data3 += [(data2[0+a:10+a])]


    df = pd.DataFrame(data3)
    df.insert(loc=0,column=11 ,value=df_names)
    df_list.append(df)
    result_df = pd.concat(df_list, ignore_index=True)
result_df.columns = col_name
print(result_df)

with open('result_df.csv', mode='w', encoding='utf-8-sig') as f:
    f.write(result_df.to_csv(index=False))


