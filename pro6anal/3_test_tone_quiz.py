import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
'''
[one-sample t 검정 : 문제1]  
영사기( 프로젝터 )에 사용되는 구형 백열전구의 수명은 250 시간이라고 알려졌다. 
한국 연구소에서 수명이 50 시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명 시간 관련 자료를 얻었다. 
한국 연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.
수집된 자료 :  305 280 296 313 287 240 259 266 318 280 325 295 315 278

귀무 가설 : 새 백열전구의 수명은 300 시간이다
대립 가설 : 새 백열전구의 수명은 300 시간이 아니다.
'''
print('='*30,'[one-sample t 검정 : 문제1]  ','='*30)
tdata1 = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
# tdata1 = pd.DataFrame(tdata1, columns='life')
print("표본의 평균 :",np.mean(tdata1))
print("표본의 크기 :",len(tdata1))

# 1. 정규성 검정
print('-'*20,'정규성 검정','-'*20)
print(stats.shapiro(tdata1)) # pvalue=0.82086
print('alpha=0.05 < pvalue=0.82086 이므로 정규성을 따른다.')
# Q-Q plot확인
# stats.probplot(tdata1, plot=plt)
# plt.show()
print()

# 2 t-test검정하기
print('-'*20,'2) t-test검정하기','-'*20)
print(stats.ttest_1samp(tdata1, 300))
# statistic=-1.5564356, pvalue=0.1436062, df=13

# 3. 결과 해석
print('-'*20,'3) 결과 해석하기','-'*20)
print('''
    해석 1 p-value)
        유의확률(α)0.05 < pvalue=4.016907430912754e-05 이므로 귀무가설 유지
        영사기( 프로젝터 )에 사용되는 구형 백열전구의 수명은 300 시간 이다.
        라는 의견이 받아들여짐
    해석 2 t분포표)
        임계값 1.771 < t통계량 -1.5564356 이므로 임계값보다 왼쪽에 위치
        귀무 채택역에 존재하므로 귀무가설을 인정한다 \n
    ''')


'''
[one-sample t 검정 : 문제2] 
국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다.
A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 
A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.  
실습 파일 : one_sample.csv
참고 : time에 공백을 제거할 땐 ***.time.replace("     ", ""),
null인 관찰값은 제거.

귀무가설 : A회사에서 생산된 노트북 평균 사용 시간이 5.2 시간이다
대립가설 : A회사에서 생산된 노트북 평균 사용 시간이 5.2 시간이 아니다.
    차이가 클 수도 작을 수도 있기 때문에 양측검정을 하는게 좋다.
'''
# 데이터 정제(전처리)
print('='*30,'[one-sample t 검정 : 문제2]  ','='*30)
tdata2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')

# 비어있는 값 결측치 처리하기
tdata2['time'] = tdata2.time.replace("     ", np.nan)

# 결측치 제거, 형변환
tdata2.dropna(subset='time', inplace=True)
tdata2['time'] = tdata2['time'].astype(float)
print(tdata2.head(3), len(tdata2))
print()

print("표본의 평균 :",np.mean(tdata2['time']))
print("표본의 크기 :",len(tdata2['time']))

# 1. 정규성 검정
print('-'*20,'정규성 검정','-'*20)
print(stats.shapiro(tdata2['time'])) # pvalue=0.7242303
print('유의확률0.05 < pvalue=0.7242303이므로 정규성을 따른다.')
print()

# histplot 확인하기
# sns.histplot(tdata2['time'], kde=True)
# Q-Q plot확인하기
# stats.probplot(tdata2['time'], plot=plt)
plt.show()
print()

# 2 t-test검정하기
print('-'*20,'2) t-test검정하기','-'*20)
print(stats.ttest_1samp(tdata2['time'], 5.2))
# statistic=3.9460, pvalue=0.0001416, df=108


# 3. 결과 해석
print('-'*20,'3) 결과 해석하기','-'*20)
print('''
    해석 1 p-value)
        유의확률(α)0.05 > pvalue=0.0001416 이므로 귀무가설 기각
        A회사에서 생산된 노트북 평균 사용 시간이 5.2 시간이 아니다.
        라는 의견이 받아들여짐
    해석 2 t분포표)
        임계값 1.645 < t통계량 3.9460 이므로 임계값보다 오른쪽에 위치
        귀무 기각역에 존재하므로 귀무가설을 기각 대립가설을 인정한다 \n
    ''')

'''
[one-sample t 검정 : 문제3] 
https://www.price.go.kr/tprice/portal/main/main.do 에서 
메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료(엑셀)를 
파일로 받아 미용 요금을 얻도록 하자. 
정부에서는 전국 평균 미용 요금이 15000원이라고 발표하였다. 
이 발표가 맞는지 검정하시오. (월별)

귀무 가설 : 전국 평균 미용 요금이 15000원이다.
대립 가설 : 전국 평균 미용 요금이 15000원이 아니다.
    클 수도 작을 수도 있다.
'''
print('='*30,'[one-sample t 검정 : 문제3]  ','='*30)
# 데이터 정제하기(전처리)
tdata3 = pd.read_csv("개인서비스지역별동향.csv")
tdata3 = tdata3.T.reset_index()
tdata3.columns =['지역','미용가격']
# print(tdata3)
tdata3.dropna(subset='미용가격', inplace=True)
tdata3 = tdata3[2:].reset_index(drop=True)
tdata3['미용가격'] = tdata3['미용가격'].astype('Int64')
# print(tdata3)
print()

print("표본의 평균 :",np.mean(tdata3['미용가격']))
print("표본의 크기 :",len(tdata3['미용가격']))
print()

# 1. 정규성 검정
print('-'*20,'정규성 검정','-'*20)
print(stats.shapiro(tdata3['미용가격'])) # pvalue=0.1340241888
print('유의확률0.05 < pvalue=0.1340241888이므로 정규성을 따른다.')
print()

# # histplot 확인하기
# sns.histplot(tdata3['미용가격'], kde=True)
# # Q-Q plot확인하기
# stats.probplot(tdata3['미용가격'], plot=plt)
# plt.show()
print()

# 2 t-test검정하기
print('-'*20,'2) t-test검정하기','-'*20)
print(stats.ttest_1samp(tdata3['미용가격'], 15000))
#statistic=7.17436278458763, pvalue=3.2057661925789937e-06, df=15
print()


# 3. 결과 해석
print('-'*20,'3) 결과 해석하기','-'*20)
print('''
    해석 1 p-value)
        유의확률(α)0.05 > pvalue=3.2057661925789937e-06 이므로 귀무가설 기각
        전국 평균 미용 요금이 15000원이 아니다라는 의견 수립.
    해석 2 t분포표)
        임계값 1.771 < t통계량 7.174362 이므로 임계값보다 오른쪽에 위치
        귀무 기각역에 존재하므로 귀무가설을 기각 대립가설을 인정한다 \n
    ''')