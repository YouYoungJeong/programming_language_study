"""
<선호도 분석>
일원카이제곱 검정
5개의 스포츠 음료에 대한 선호도에 차이가 있는지 검정하기
변수 : 스포츠 음료에 대한 선호도 - 일원카이제곱
수식은 같으나 이론을 펼칠때 선호하냐 하지 않냐의 차이
귀무가설 : 기대치와 관찰치는 차이가 없다.스포츠 음료의 선호도에 차이가 없다.
대립가설 : 기대치와 관찰치는 차이가 있다.스포츠 음료의 선호도에 차이가 있다.
"""
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinkdata.csv")
print(data)

# 일원카이제곱
print(stats.chisquare(data['관측도수']))
# statistic=20.489, pvalue=0.00039991
"""
pvalue값이 0에 가까울수록 카이제곱(통계량이) 커진다.
통계량이 커질 수록 pvaluer값이 작아지므로 서로 반비례(Trade-off) 관계
통계량이 커질 수록 귀무가설을 기각할 확률이 커짐
"""

# 기대 빈도 구하기
exp = [data['관측도수'].sum() / 5] * 5
print(exp) # 기대빈도 : 50.8

stat, p=  stats.chisquare(f_obs=data['관측도수'], f_exp=exp)
print(f"stat: {stat} \np={p}")
# print(f"\ndof={dof} \nexpected={expected}") # 이원카이제곱이 아니라 반환하지 않는 값.

print("""
    판정 : 유의수준 0.05 > p-value 0.0003이므로 귀무 기각
    대립가설인 스포츠 음료의 선호도에 차이가 있다. 라는 의견이 받아들여짐.
    """)

# 시각화하기
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 기대도수
total = data['관측도수'].sum()
expected = [total/len(data)] * len(data)
x = np.arange(len(data))    # 음료 갯수만큼 x축 좌표 잡아주기.
width = 0.35                # 막대그래프 넓이 주기 

# 그리기
plt.figure(figsize=(10, 5))
plt.bar(x - width / 2, data['관측도수'], width=width, label='관측도수')
plt.bar(x + width / 2, expected, width=width, label='기대도수', alpha=0.6)
plt.xticks(x, data['음료종류'])
plt.xlabel('음료종류')
plt.ylabel('도수')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 보고서 작성하기
print("카이제곱 검정 결과와 그래프를 근거로 어떤 음료가 더 인기 있는지를 분석")
data['기대도수'] = expected
data['차이(관측-기대)'] = data['관측도수']-data['기대도수']
data['차이비율(%)'] = round(data['차이(관측-기대)']/ expected * 100, 2)
# print(data)
# 차이가 많은순으로 정렬
data.sort_values(by='차이(관측-기대)',ascending=False, inplace=True)
# 기존 인덱스 지우고 새로운 인덱스 설정
data.reset_index(drop=True, inplace=True)
print(data)
