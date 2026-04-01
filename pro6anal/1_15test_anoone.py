'''
일원분산분석(One way ANOVA)으로 평균 차이 검정

예제 ) 강남구에 있는 GS편의점 3개 지역 알바생의 급여에 대한 평균차이 검정하기
    요인 : GS편의점

귀무가설 : GS편의점 3개 지역 알바생의 급여에 대한 평균 차이가 없다.
대립가설 : GS편의점 3개 지역 알바생의 급여에 대한 평균 차이가 있다. 
'''
'''
참고
    anova_lm() : 정규성, 등분산성이 깨지면 p-value 신뢰 불가
    f_oneway() : 정규성 깨지면 Kruskal-Wallis - stats.kruskal() 사용, 
                등분산성이 깨지면 welch ANOVA 사용
'''

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import urllib.request


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#읽어오기 1) DataFrame으로 읽어오기
uri = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3.txt"
data = pd.read_csv(uri, header=None) # txt파일이지만 ,로 구분되어 있으므로 csv로 읽어옴
print(type(data)) # <class 'pandas.core.frame.DataFrame'>
# ndarray로 형변환 하기
# data = data.values

# 읽어오기 2) - 배열로 읽어오기 (웹에서 읽어오기 때문에 import urllib.request 필요)
data = np.genfromtxt(urllib.request.urlopen(uri), delimiter=",")
print(data, type(data)) # <class 'numpy.ndarray'>
print(data.shape) # (22, 2)

# 세개의 집단에 월급자료 얻기, 평균
group1 = data[data[:, 1] == 1, 0]
group2 = data[data[:, 1] == 2, 0]
group3 = data[data[:, 1] == 3, 0]
print("group1 data : ", group1, "/ group1 mean : ", np.mean(group1)) # 316.625
print("group2 data : ", group2, "/ group2 mean : ", np.mean(group2)) # 256.444
print("group3 data : ", group3, "/ group3 mean : ", np.mean(group3)) # 278.0
print()

# 정규성 확인
print("group1-shapiro-pvalue : ",stats.shapiro(group1).pvalue)
print("group2-shapiro-pvalue : ",stats.shapiro(group2).pvalue)
print("group3-shapiro-pvalue : ",stats.shapiro(group3).pvalue)
print(" 전부다 정규성 만족한다 \n")

# 등분산성 확인 (bartlett -표본의 차이가 클때 사용)
print("등분산성 확인 : ",stats.levene(group1, group2, group3).pvalue)   
# 0.04584 - 반올림하면 0.05이므로 등분산성 만족이라고 할 수 있다
print("등분산성 확인 : ",stats.bartlett(group1, group2, group3).pvalue) # 0.35080
print()

# 데이터의 퍼짐정도 시각화
# plt.boxplot([group1, group2, group3], showmeans=True)
# plt.show()

# 일원분산분석 방법 1 - anova_lm()
# 보고서 만들때(sum_sq mean_sq이 필요하기 때문에 사용함.)
print("일원분산분석 방법 1 - anova_lm()")
df = pd.DataFrame(data=data, columns=['pay','group'])
print(df)
# C()로 둘러주면 group은 범주형 이라는 뜻
lm_model = ols('pay ~ C(group)', data=df).fit()
print(anova_lm(lm_model, typ=1))
print("해석 : p = 0.043589 < 0.05 α 이므로 귀무 기각.")
print()

# 일원분산분석 방법 2 - f_oneway() : 결과는 동일
# 단순하게 결과만 필요한 경우 사용.
print("일원분산분석 방법 2 - f_oneway()")
f_stat, p_val = stats.f_oneway(group1, group2, group3)
print("f : ",f_stat, "p :",p_val)
# f :  3.711335988266977 p : 0.043589334959178244
print()

# 사후 검정 하기
print("사후 검정 하기")
tukResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(tukResult)

# 사후 검정 시각화하기
tukResult.plot_simultaneous(xlabel="mean", ylabel="group")
plt.show()

# reject 에서 1그룹과 2그룹의 평균의 차이가 있다라고 나옴.
