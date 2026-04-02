'''
이원분산분석(two-way ANOVA)
요인이 복수 - 각 요인의 레벨(요인 안에 있는 그룹, 집단)도 복수
두개의 요인에 대한 집단(독립변수) 각각이 종속변수(평균)에 영향을 주는지 검정
주효과(=개별효과) : 독립변수들이 각각 독립적으로 종속변수에 미치는 영향을 검정함
상호작용효과(=교호작용:Interaction Effect)
    독립변수들이 서로 연관되어 종속변수에 미치는 영향을 검정함
    한 독립변수가 종속변수에 미치는 영향이 다른 독립변수의 수준에 따라 달라지는 현상

실습1: 태아 수와 관측자 수가 태아의 머리둘레 평균에 영향을 주는가? (group3_2.txt)
주효과 가설
    요인1 - 태아 수)
        귀무가설 : 태아 수와 태아의 머리둘레 평균은 차이가 없다.
        대립가설 : 태아 수와 태아의 머리둘레 평균은 차이가 있다.
    요인2 - 관측자 수)
        귀무가설 : 관측자 수와 태아의 머리둘레 평균은 차이가 없다.
        대립가설 : 관측자 수와 태아의 머리둘레 평균은 차이가 있다.
교호작용 가설
    귀무가설 : 교호작용이 없다. - 태아수와 관측자수는 관련이 없다.
    대립가설 : 교호작용이 있다. - 태아수와 관측자수는 관련이 있다.
'''
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
from statsmodels.formula.api import ols
# one way만 존재하고 two way는 anova_lm를 이용해 이원분석을 해야한다.
from statsmodels.stats.anova import anova_lm
np.set_printoptions(suppress=True, precision=10)
pd.options.display.float_format = '{:.10f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3_2.txt")
print(data.head(3), data.shape)    # (36, 3)
print(data['태아수'].unique())     # [1 2 3]
print(data['관측자수'].unique())   # [1 2 3 4]

# 시각화하기
# data.boxplot(column='머리둘레', by='태아수')
# plt.show()
# data.boxplot(column='머리둘레', by='관측자수')
# plt.show()

# ols 모델 만들기
linmodel = ols("머리둘레 ~ C(태아수) + C(관측자수)", data=data).fit() # 교호작용이 없는거

linmodel = ols("머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수)", data=data).fit() 
# + C(태아수):C(관측자수) : 교호작용을 확인하겠다.라는 의미
linmodel = ols("머리둘레 ~ C(태아수) * C(관측자수)", data=data).fit() 
# C(태아수) * C(관측자수) : 두개의 요인을 곱하면 교호작용을 확인하겠다.라는 의미
result = anova_lm(linmodel, typ=2)
print(result)
'''
                        sum_sq    df          F        PR(>F)
C(태아수)               324.008889 2.0  2113.101449  0.0000000000 : 귀무기각
C(관측자수)             1.198611   3.0     5.211353  0.0064970547 : 귀무기각
C(태아수):C(관측자수)   0.562222   6.0     1.222222  0.3295509350
Residual                1.840000  24.0       NaN           NaN

->  태아수 PR(>F):0.00000 < α:0.05 이므로 귀무가설을 기각
    태아 수와 태아의 머리둘레 평균은 차이가 있다라는 가설 채택 - 유의하다.
->  관측자수 PR(>F):0.0064970547 < α:0.05 이므로 귀무가설을 기각한다
    관측자수에 따라 태아의 머리둘레 평균은 차이가 있다라는 가설 채택 - 유의하다.
-> C(태아수):C(관측자수) - 교호작용(상호작용)
    PR(>F):0.329550 > α:0.05 이므로 귀무가설을 채택함.
    교호작용이 없다. - 태아수와 관측자수는 관련이 없다.-유의하지않다.
-> 해석 :   태아수와 관측자수는 각각 종속변수에 유의한 영향을 미친다.
            그러나 태아수와 관측자수간의 상호작용(교호작용)효과는 유의하지 않다.
            즉, 주효과는 있고 상호작용은 없음.
'''

# ===========================================================================
'''
실습2: poison(독)과 treat(응급처치)가 독 퍼짐 시간의 평균에 영향을 주는가? (poison_treat.csv)
주효과 가설
    요인1 - poison)
        귀무가설 : poison 종류와 독 퍼짐 시간의평균은 차이가 없다.
        대립가설 : poison 종류와 독 퍼짐 시간의평균은 차이가 있다.
    요인2 - treat)
        귀무가설 : treat방법과 독퍼짐 시간의 평균 차이가 없다.
        대립가설 : treat방법과 독퍼짐 시간의 평균 차이가 있다.
교호작용 가설
    귀무가설 : 교호작용이 없다. - poison 종류와 treat방법이 관련이 없다.
    대립가설 : 교호작용이 있다. - poison 종류와 treat방법이 관련이 있다.
'''

data2 = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/poison_treat.csv",
                    index_col=0)
print(data2.head(3), data2.shape) # (48, 3)

# 표본수 확인하기
print(data2.groupby('poison').agg(len))
print(data2.groupby('treat').agg(len))
print(data2.groupby(['poison','treat']).agg(len)) 
# 요인별 레벨의 표본수는 4로 동일(모든 집단별 표본 수가 동일하므로 - 균형적인 설계(균형설계)가 되어있다.)
linmodes2 = ols("time ~ C(poison) * C(treat)", data=data2).fit()
print(anova_lm(linmodes2))
'''
                    df    sum_sq   mean_sq          F        PR(>F)
C(poison)            2.0  1.033013  0.516506  23.221737  0.0000003331 : 귀무기각
C(treat)             3.0  0.921206  0.307069  13.805582  0.0000037773 : 귀무기각
C(poison):C(treat)   6.0  0.250138  0.041690   1.874333  0.1122506083 : 귀무채택

->  poison PR(>F):0.0000003331 < α:0.05 이므로 귀무가설을 기각
    poison 종류와 독 퍼짐 시간의평균은 차이가 있다. 가설 채택 - 유의하다.
->  treat PR(>F):0.0000037773 < α:0.05 이므로 귀무가설을 기각한다
    treat방법과 독퍼짐 시간의 평균 차이가 있다 라는 가설 채택 - 유의하다.
-> C(poison):C(treat) - 교호작용(상호작용)
    PR(>F):0.1122506083 > α:0.05 이므로 귀무가설을 채택함.
    교호작용이 없다. - poison 종류와 treat방법이 관련이 없다.-유의하지않다.
'''

# 사후 분석
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tkuresult1 = pairwise_tukeyhsd(endog=data2.time, groups=data2.poison)
print(tkuresult1)
'''
Multiple Comparison of Means - Tukey HSD, FWER=0.05 
====================================================
group1 group2 meandiff p-adj   lower   upper  reject
----------------------------------------------------
    1      2  -0.0731 0.5882 -0.2525  0.1063  False
    1      3  -0.3412 0.0001 -0.5206 -0.1619   True
    2      3  -0.2681 0.0021 -0.4475 -0.0887   True
----------------------------------------------------
'''
tkuresult2 = pairwise_tukeyhsd(endog=data2.time, groups=data2.treat)
print(tkuresult2)
'''
Multiple Comparison of Means - Tukey HSD, FWER=0.05 
====================================================
group1 group2 meandiff p-adj   lower   upper  reject
----------------------------------------------------
    A      B   0.3625  0.001  0.1253  0.5997   True
    A      C   0.0783 0.8143 -0.1589  0.3156  False
    A      D     0.22 0.0778 -0.0172  0.4572  False
    B      C  -0.2842 0.0132 -0.5214 -0.0469   True
    B      D  -0.1425  0.387 -0.3797  0.0947  False
    C      D   0.1417 0.3922 -0.0956  0.3789  False
----------------------------------------------------
'''

# 사후분석 시각화
tkuresult1.plot_simultaneous(xlabel='mean of time', ylabel='posion')
tkuresult2.plot_simultaneous(xlabel='mean of treat', ylabel='treat')
plt.show()
plt.close()