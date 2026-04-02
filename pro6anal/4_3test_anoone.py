'''
일원분산분석(One way ANOVA)

데이터셋 설명
1) tsales.csv(어느 식당의 판매 매출 기록) - 시계열 데이터
    YMD:년월일	AMT:매출액	CNT:판매량

2) tweather.csv(날씨정보) - 시계열 데이터
    stnId	tm:날짜	avgTa:평균기온	minTa:최저기온	maxTa:최대기온	sumRn:강수량	
        maxWs	avgWs	ddMes

어느 음식점의 매출 데이터와 기상청이 제공한 날씨 데이터를 활용하여 
최고온도에 따른 매출액의 평균에 차이가 있는지 검정
세 집단(범주형) : 추움, 보통, 더움

귀무 가설 : 어느 음식점의 매출 데이터는 온도에 따라 매출액 평균에 차이가 없다.
대립 가설 : 어느 음식점의 매출 데이터는 온도에 따라 매출액 평균에 차이가 있다.
'''
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# numpy 배열 출력 옵션
np.set_printoptions(suppress=True, precision=10)
# pandas DataFrame / Series 출력 옵션
pd.options.display.float_format = '{:.10f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print('-'*20,' 데이터 전처리 ','-'*20)
# 둘이 같이 들고있는건 날짜뿐인데 서로의 타입이 다르다.
# int를 들고올때 타입 변환해서 들고오기 dtype={"YMD":"object"}

# 매출 데이터 읽기
'''
        YMD    AMT  CNT
0  20190514      0    1
1  20190519  18000    1
'''
print("매출 데이터 읽기")
sales_data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv",
                        dtype={"YMD":"object"}) # int -> object로 변환해서 읽기
print(sales_data.head(2))
# 합치려면 type을 잘봐야한다.
print(sales_data.info())   # 328 * 3 
print()

# 날씨 데이터 읽기
'''
    stnId          tm  avgTa  minTa  maxTa  sumRn  maxWs  avgWs  ddMes
0    108  2018-06-01   23.8   17.5   30.2    0.0    4.3    1.9    0.0
1    108  2018-06-02   23.4   17.6   30.1    0.0    4.5    2.0    0.0
'''
print("날씨 데이터 읽기")
wt_data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv")
print(wt_data.head(2))
print(wt_data.info()) # 702 * 9
print()

# 날짜데이터 형태 맞추기
print("날짜데이터 형태 맞추기")
#sales: YMD 20190514, wt: tm 2018-06-01 데이터 병합을위해 데이터 변환 필요
wt_data['tm'] = wt_data.tm.map(lambda x:x.replace("-",""))
print(wt_data.head(2)) # 20180601 모양이 바뀌었다
print()

# 두 데이터 병합하기
print("두 데이터 병합하기")
# 기준! sales_data로 YMD,tm이 일치하는 행만 제외하고 나머지는 다 없어짐.
frame = sales_data.merge(wt_data, how='left', left_on='YMD', right_on="tm")
print(frame.head(3),"\n",len(frame)) #  328 * 12
print()

# 필요한 데이터만 가져오기
print("필요한 데이터만 가져오기")
data = frame.iloc[:,[0, 1, 7]] # ['YMD', 'AMT', 'maxTa']
print(data.head(3))

# 결측치 확인
print("결측치 확인")
print(data.isnull().sum())
print()

# 데이터 확인
print(data.maxTa.describe())
# plt.boxplot(data.maxTa)
# plt.show()

# 온도를 세 그룹으로 분리(연속형(int) -> 범주형)
print(data.isnull().sum())
data['ta_gubun'] = pd.cut(data.maxTa, bins=[-5, 8, 24, 37], labels=[0, 1, 2])
print(data.head(), '' , data['ta_gubun'].unique())
print()
'''
        YMD     AMT  maxTa ta_gubun
0  20190514       0   26.9        2
1  20190519   18000   21.6        1
2  20190521   50000   23.8        1
3  20190522  125000   26.5        2
4  20190523  222500   29.2        2 
Categories (3, int64): [0 < 1 < 2]
'''
# 데이터 그룹화(범주화) 데이터 나누기
print("데이터 그룹화(범주화) 데이터 나누기")
x1 = np.array(data[data['ta_gubun'] == 0].AMT) # 추운 그룹 
x2 = np.array(data[data['ta_gubun'] == 1].AMT) # 보통
x3 = np.array(data[data['ta_gubun'] == 2].AMT) # 더움
print(x1[:5], x2[:5], x3[:5])
print()

# 정규성 확인하기 - 만족O
print('정규성 확인하기')
print(stats.shapiro(x1).pvalue) # 0.248192
print(stats.shapiro(x2).pvalue) # 0.038825
print(stats.shapiro(x3).pvalue) # 0.318298
print()

# 등분산성 확인하기 - 만족X
print('등분산성 확인하기')
print("levene-pvalue :", stats.levene(x1, x2, x3).pvalue) # 0.0390
print("bartlett-pvalue :",stats.bartlett(x1, x2, x3).pvalue) # 0.0096775
print()

# 온도별 매출액 평균
print("온도별 매출액 평균")
spp = data.loc[:,['AMT','ta_gubun']]
print(spp.groupby("ta_gubun").mean())

print(np.mean(x1)) # 1032362.3188405797
print(np.mean(x2)) # 818106.8702290077
print(np.mean(x3)) # 553710.9375

# 시각화 하기
group1 = x1
group2 = x2
group3 = x3
# plt.boxplot([group1, group2, group3], showmeans=True)
# plt.show()

# 검정하기 - f_oneway
print("검정하기 - f_oneway")
print(stats.f_oneway(group1, group2, group3))
# statistic=99.190801, pvalue=2.3607371e-34
print("해석 : pvalue:2.3607371e-34 < α:0,05 이므로 귀무기각, 대립인정" \
"\n어느 음식점의 매출 데이터는 온도에 따라 매출액 평균에 차이가 있다.라는 의견 인정\n")

# 정규성 깨지면 Kruskal-Wallis - stats.kruskal() 사용, 등분산성이 깨지면 welch ANOVA 사용
print("정규성 깨지면 Kruskal-Wallis - stats.kruskal() 사용")
print(stats.kruskal(group1, group2, group3))
#KruskalResult:  statistic=132.702259, pvalue=1.52781e-29 - 귀무 기각 
print()

print("등분산성이 깨지면 welch ANOVA 사용")
# 모듈 설치 필요 : pip install pingouin 
from pingouin import welch_anova
print(welch_anova(dv="AMT", between='ta_gubun', data=data))
'''
    Source   ddof1          ddof2              F        p_unc          np2
0  ta_gubun   2    189.6514001796   122.2212420883  0.0000000000  0.3790381654
'''
# p-value = 0.0000000000 귀무기각.

# 사후 검정 하기
print("사후 검정 하기")
tukResult = pairwise_tukeyhsd(endog=spp.AMT, groups=spp.ta_gubun, alpha=0.05)
print(tukResult)

# 사후 검정 시각화하기
tukResult.plot_simultaneous(xlabel="mean", ylabel="group")
plt.show()
# 세개의 구분이 전부 겹치는곳없다.  reject이 전부 True