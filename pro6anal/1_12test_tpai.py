'''
paired samples t-test(대응표본 t검정, 동일집단표본 t검정, 쌍체 t검정)
서로 대응인 두 집단의 평균 차이 검정
하나의 집단에 대해 독립변수를 적용하기 전과 후의 종속변수(평균)의 수준을 측정하고 
평균의 차이가 통계적으로 유의한지를 분석하는 방법
동일한 관찰 대상으로부터 처리 이전과 처리 이후를 1:1로 대응시킨 검정 방법
두 집단으로 부터의 표본을 대응표본(paired sample)이라고 한다.
집단 간 비교가 아니므로 등분산 검정을 할 필요가 없다.
ex) 광고 전후의 상품선호도 측정, 투자 대비 상품 판매량...

실습 1) 3강의실 학생들을 대상으로 특강이 시험점수에 영향을 주었는가?
귀무가설: 특강 전후에 시험 점수는 차이가 없다
대립가설: 특강 전후에 시험 점수는 차이가 있다
'''
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(123) 
x1 = np.random.normal(75, 10, 100) # normal = 정규 분포를 따름
x2 = np.random.normal(80, 10, 100)

# 정규성 확인
print('-'*20,' 정규성 검정 ','-'*20)
# 그래프로 확인
sns.displot(x1, kde=True)
sns.displot(x2, kde=True)
plt.show()
# shapiro로 확인
print(stats.shapiro(x1).pvalue) # 0.27487044002059957
print(stats.shapiro(x2).pvalue) # 0.10214157305396582
print("둘다 유의확률(0.05)보다 크기 때문에 정규성을 만족함.\n")

# 대응표본 t검정
print('-'*20,'paired samples t-test','-'*20)
print(stats.ttest_rel(x1, x2))
# statistic=-3.00310270, pvalue=0.0033837, df=99
print('''
    해석 : p-value:0.0033 < 유의확률:0.05 이므로 귀무가설 기각
        특강 전후에 시험 점수에 영향을 주었다. 라는 의견을 받아 들임.
    ''')