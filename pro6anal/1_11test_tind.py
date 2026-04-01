'''
어느 음식점의 매출 데이터와 기상청이 제공한 날씨 데이터를 활용하여 
강수여부에 따른 매출액의 평균에 차이가 있는지 검정

데이터셋 설명
1) tsales.csv(어느 식당의 판매 매출 기록) - 시계열 데이터
    YMD:년월일	AMT:매출액	CNT:판매량

2) tweather.csv(날씨정보) - 시계열 데이터
    stnId	tm:날짜	avgTa:평균기온	minTa:최저기온	maxTa:최대기온	sumRn:강수량	
        maxWs	avgWs	ddMes

두 집단(범주형) : 비(눈)가 올때(=강수량有), 비(눈)가 안올때(=맑을때)

귀무 가설 : 어느 음식점의 매출 데이터는 강수 여부에 따라 매출액 평균에 차이가 없다.
대립 가설 : 어느 음식점의 매출 데이터는 강수 여부에 따라 매출액 평균에 차이가 있다.
'''
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
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
data = frame.iloc[:,[0, 1, 7, 8]] # ['YMD', 'AMT', 'maxTa', 'sumRn']
print(data.head(3))

# 결측치 확인
print("결측치 확인")
print(data.isnull().sum())
print()

# 강수량의 유무 확인
print("강수량의 유무 확인")
print((data['sumRn'] > 0).sum()) # 강수량이 조금이라도 있으면 True
print()

# 칼럼추가 : 강수량이 있으면 1 없으면 0
print("칼럼추가")
# 방법1
# data['rain_yn'] = (data['sumRn'] > 0).astype(int)

# 방법2
# print(True * 1, " ", False * 1) # 이러면 숫자화 되서 .astype(int) 안해도 된다.
data['rain_yn'] = (data.loc[:,('sumRn')] > 0) * 1
print(data.head())
print()
'''
        YMD     AMT  maxTa  sumRn  rain_yn
0  20190514       0   26.9    0.0        0
1  20190519   18000   21.6   22.0        1
2  20190521   50000   23.8    0.0        0
-> 원하는 데이터셋이 나옴
'''

# box plot 시각화
print("box plot 시각화 전처리 ")
# 시각화하기전 데이터 추출
sp = np.array(data.iloc[:, [1, 4]]) # AMT, rain_yn만 추출 (2차원배열)
# print(sp)
'''
[[      0       0] # 0번째열:AMT, 1번째열:rain_yn
[  18000       1]
'''
# 비의 유무에 따른 매출액 추출하기
tg1 = sp[sp[:, 1] == 0, 0] # 집단 1 : 비 안올때 매출액
tg2 = sp[sp[:, 1] == 1, 0] # 집단 2 : 비 올때 매출액
print(tg1[:3], tg2[:3])

print("box plot 확인하기\n")
# 매출액의 분포를 보면 평균분포도 대략 확인이 가능하다
plt.boxplot([tg1, tg2], meanline=True, showmeans=True, notch=True)
plt.show()

# boxplot으로 확인한 결과 잘 안보임.
print("날씨별 평균 매출액 확인하기")
print("tg1(맑은날) 매출액 평균 :",np.mean(tg1)) # 761040.25
print("tg2(비온날) 매출액 평균 :",np.mean(tg2)) # 757331.52
print()

# =============== 독립표본 t검정 시작 =======================
print('-'*20,' 정규성 검정 ','-'*20)
# 정규성 검정
print(len(tg1), " ", len(tg2),"\n") # 236   92 - 비오는 날의 데이터가 양이 적다.
print(stats.shapiro(tg1).pvalue) # 0.05605
print(stats.shapiro(tg2).pvalue) # 0.88275
print("두 그룹다 유의수준(0.05)보다 크므로 정규성을 만족한다\n")

print('-'*20,' 등분산성 검정 ','-'*20)
# 등분산성 검정
print(stats.levene(tg1, tg2).pvalue,"\n") # 0.7123452333011173
print("유의수준(0.05)보다 크므로 등분산성을 만족하므로 t검정 사용 가능\n")

print('-'*20,'independent two samples t-test','-'*20)
# t-test
print(stats.ttest_ind(tg1, tg2, equal_var=True))
# statistic=0.101098, pvalue=0.919534, df=326
print('''  
        해석 : 정규성, 등분산성 존건은 충족함
        p-value:0.919534 > 유의수준:0.05 이므로 귀무가설 채택함
        최종적으로 매출 데이터는 강수 여부에 따라 매출액 평균에 차이가 없다라는 의견 유지
    ''') 
# 후속조치가 필요하다.