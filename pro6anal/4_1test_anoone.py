'''
ANalysis Of Variance(ANOVA)
세 개 이상의 모집단에 대한 가설검정 
분산분석이라는 용어는 분산이 발생한 과정을 분석하여 요인에 의한 분산과 
요인을 통해 나누어진 각 집단 내의 분산으로 나누고 요인에 의한 분산이 
의미 있는 크기를 가지는지를 검정하는 것을 의미한다.
세 집단 이상의 평균비교에서는 독립인 두 집단의 평균 비교를 반복하여 실시할 경우에 
제1종 오류가 증가하게 되어 문제가 발생한다.
이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of Variance)을 이용하게 된다.
분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 나눠진 각 집단내의 분산으로 나누고,
요인에 의한 분산이 의미 있는 크기를 가지는지를 검정한다.
아노바는 사후 분석이 필수다.

F-value = 집단간분산 / 집단내분산
따라서 의미가 있으려면 집단내분산차이보다 집단간분산차이가커야한다.

* 서로 독립인 세 집단의 평균 차이 검정
실습) 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 대상으로 실기시험을 실시. 
three_sample.csv'
일원분산분석(One way ANOVA)
독립변수 (범주형 x) : 한개의 요인:교육방법, 방법의 종류가 3가지 - 그룹이 3개
종속변수 (연속형 y) : 실기시험 평균점수 

귀무가설 : 세 가지 교육 방법에 따른 시험 점수에 차이가 없다.
대립가설 : 세 가지 교육 방법에 따른 시험 점수에 차이가 있다.
'''
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
# (선형회귀사용을 위한 모듈) 추정 및 검정, 회귀, 시계열 분석 등의 기능 제공lib
# ols(Ordinary Least Squares) - 최소제곱법
from statsmodels.formula.api import ols 


data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/three_sample.csv")
print(data.head(3))
print(len(data))
print(data.describe()) # 이상치 : score= 500.000000 발견

# 이상치(outlier) 처리
# 시각화 - 2개 발견
# plt.boxplot(data["score"])
# plt.show()

# 이상치 제거
data = data.query("score <= 100")
print(len(data))
print(data.describe())

# 교차표로 확인하기 (교육 방법별 건수) - 참고
data2 = pd.crosstab(index=data['method'], columns='count')
data2.index = ['방법1','방법2','방법3']
print(data2)

# 교차표로 확인하기 (교육 방법별 만족건수) - 참고
data3 = pd.crosstab(data['method'], data['survey'])
data3.index = ['방법1','방법2','방법3']
data3.columns=['만족','불만족']
print(data3)

# 아노바 방식 1) 회귀분석에 의한 결과값 anova_lm을 사용하려면 선형회귀모델이 필요함.
# F 통계량을 얻기위해 회귀분석 결과(linear model)를 사용함
print('아노바 방식 1) 회귀분석에 의한 결과값 - anova_lm')
import statsmodels.api as sm
# 회귀분석 모델 생성
lin_model = ols("data['score'] ~ data['method']", data=data).fit()
result = sm.stats.anova_lm(lin_model, typ=1)
print(result)
'''
mean_sq(제곱평균) : 제곱합 / 자유도
F값(검정통계량) = data['method'].mean_sq / Residual.mean_sq
p-value = f통계를 이용해서 구해진 p값
            df(자유도) | sum_sq(제곱합) | mean_sq(제곱평균) | F(F값)   |PR(>F) - f통계를 이용해서 구해진 p값
data['method']   1.0     27.980888      27.980888           0.122228    0.727597
Residual(잔차)  76.0    17398.134497    228.922822          NaN          NaN
-> 결과를 해석해야함.
'''
f_value = result.loc["data['method']","F"]
p_value = result.loc["data['method']","PR(>F)"]
print("f_value : ", f_value)
print('p_value : ', p_value)

print(''' 
    해석 : p-value 가 0.727597이므로 α보다 크기 때문에 귀무채택력 안에 있다.
        따라서 세 가지 교육 방법에 따른 시험 점수에 차이가 없다.의견 유지
    ''')
print()

# 사후분석(Post Hoc Analysis) 하기
print("사후분석(Post Hoc Analysis) 하기")
'''
세 가지 교육 방법에 따른 시험 점수에 차이여부는 알려주지만 
정확히 어느 그룹의 평균값이 의미가 있는지는 알려주지는 않는다.
즉 그룹간 평균 차이를 구체적으로 알려주지 않는다. 
그러므로 그룹 간의 관계를 보기 위해 추가적인 사후분석(Post Hoc Analysis)이 필요하다.'''
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukresult = pairwise_tukeyhsd(endog=data['score'], groups=data['method'])
print(tukresult)

# pairwise_tukeyhsd 결과표
'''  
Tukey HSD : 원래 반복 수가 동일하다는 가정하에 고안된 방법
장점 : 집단 간 평균 차이를 정밀하게 확인 가능
단점 : 집단의 표본 수의 차이가 크면 결과의 신뢰도 떨어짐

reject에서 평균차이를 보는것임 
    유의미한 차이가 없으면 False, 유의미한 차이가 있으면 True

Multiple Comparison of Means - Tukey HSD, FWER=0.05 
====================================================
group1 group2 meandiff p-adj   lower   upper  reject
----------------------------------------------------
    1      2   0.9725 0.9702 -8.9458 10.8909  False
    1      3   1.4904 0.9363 -8.8183  11.799  False
    2      3   0.5179 0.9918 -9.6125 10.6483  False
----------------------------------------------------
-> reject 전부다 False이므로 평균의 차이가 없다.
'''

# Tukey HSD 시각화
tukresult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
print("평균의 차이가 크면 선그래프의 위치가 다 다를것임 지금은 전부 겹쳐있다. 그래서 의미가 없다라고 나옴.")