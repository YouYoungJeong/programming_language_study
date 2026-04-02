'''
independent two samples t-test : 독립표본 t검정
(나중에 그룹화(Classification, Clustering)를 위해 검증방법으로 아주 중요함.)

실습) 두 가지 교육방법에 따른 평균시험 점수에 대한 검정 수행 two_sample.csv'
    교육방법 method 

귀무가설 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 없다.
대립가설 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 있다.
'''
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from scipy.stats import levene, bartlett, fligner

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/two_sample.csv")
print(data.head())
print()

print('-'*20,' 데이터 전처리 ','-'*20)
# nan값 확인 - 데이터의 양이 얼마 없을땐 mean으로 주는것도 좋다.
# print(data['score'].isnull().sum())
# print(data.isnull().any)

# 교육 방법별 점수로 데이터 분리.
print("교육 방법별 점수로 데이터 분리")
ms = data[['method', 'score']]
m1 = ms[ms['method']==1]    # method 1
m2 = ms[ms['method']==2]    # method 2
print(m1.head(3))
print(m2.head(3))
print()

# 교육방법에서 score만 별도 추출 후 nan값 처리(fillna)
print("교육방법에서 score만 별도 추출")
score1 = m1['score']
score2 = m2['score']
print(score1.isnull().sum())    # 0
print(score2.isnull().sum())    # 2
score2 = score2.fillna(score2.mean())   #  NaN을 평균으로 대체
print()

print('-'*20,' 정규성 검정 ','-'*20)
print(stats.shapiro(score1)) # 0.36799
print(stats.shapiro(score2)) # 0.67142
print("두그룹 전부 정규성을 만족")
print()

# 정규성 시각화
sns.histplot(score1, kde=True)
sns.histplot(score2, kde=True, color='blue')
plt.show()

print('-'*20,' 등분산성 검정 ','-'*20)
print("등분산성(한번에 보기)p-value: ",levene(score1, score2).pvalue) # 0.4568427112977608
print("등분산성 만족")
print()

print('-'*20,' t-test 검정 ','-'*20)
result = stats.ttest_ind(score1, score2, equal_var=True)
print(result)
# statistic=-0.196493, pvalue=0.845053, df=48
print('''
    검정결과)
        pvalue=0.845053 > 유의수준0.05 이므로 이 데이터는 우연히 발생했다 보고 귀무가설 채택
        두 가지 교육방법에 따른 평균시험 점수에 차이가 없다.라는 의견 유지
    ''')
