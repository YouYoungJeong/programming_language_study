"""
https://finance.naver.com/marketindex/
실시간 변동데이터 읽어오기.
네이버- 증권- 시장지표 - 환율값 출력 (일정 시간에 한번씩 주기적으로 읽기)
"""
import requests
from bs4 import BeautifulSoup
import time
import sys 

sys.stdout.reconfigure(encoding='utf-8') # content사용시 바이너리 데이터를 가져옴, 그럴때 사용
url = "https://finance.naver.com/marketindex/"
headers = {"User-Agent" : "Mozilla/5.0"} 

# 무한 루프에 넣기
while True:
    res = requests.get(url=url, headers=headers)
    soup = BeautifulSoup(res.content, 'html.parser')

    # 나라 이름 가져오기
    nation = soup.select_one("h3.h_lst span.blind").get_text(strip=True)
    # print(nation) # 미국 USD

    # 환율값 가져오기 - div안에 class값이 value밖에 없어서 class로 가져옴
    price = soup.select_one(".value").get_text(strip=True)
    # print(price)

    # 원 가져오기
    unit = soup.select_one(".txt_krw .blind").get_text(strip=True)
    # print(unit) # 원

    # 변동값 가져오기
    change = soup.select_one(".change").get_text(strip=True)
    # print(change) 

    # 상승 하락 가져오기
    updown = soup.select("div.head_info.point_up span.blind")[-1].get_text(strip=True)
    # print(updown)

    # 1회만 실행하다 끝남. -> while안에 넣어서 무한루프 생성
    print(f"{nation.replace(" ","")} {price}{unit} {updown} {change}")
    
    # 시간 텀 꼭 주기! 안 그러면 차단 당함.
    time.sleep(5)