'''
회귀분석 모형의 적절성을 위한 선행조건 5가지
    선형회귀분석 모형의 적절성을 위한 선행조건
        1)정규성
        2)선형성 : 독립변수와 종속변수간에 선형형태로 적절하게 모델링 되었는지 검정
                독립변수의 변화에 종속변수도 변화하나 특정한 패턴이 있으면 X
        3)등분산성

    다중회귀분석 모형의 적절성을 위한 선행조건
        1)+2)+3)
        4)독립성
        5)다중공선성

잔차(Residual)
    제 관측값(y)과 모델이 예측한 값(y^)의 차이(y - y^)를 의미,
    모델이 데이터를 얼마나 잘 설명하는지 보여주는 척도, 모델의 적합성(등분산성, 정규성)을 진단
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import statsmodels.formula.api as smf


# 데이터 가져오기
advdf =  pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv",
                    usecols=[1,2,3,4]) # 사용하고 싶은 칼럼만 가져오기
# print(advdf.head(3)) # 확인
# print(advdf.shape) # (200, 5)
# print(advdf.info()) # float64, nullX
# print(advdf.corr())
print(advdf.corr()['sales'].sort_values(ascending=False).values) 
# tv:0.78222442 radio:0.57622257 newspaper: 0.22829903
print()

# 단순선형회귀 모델 만들기 ols
# x :tv, y:sales
lm = smf.ols(formula='sales ~ tv', data=advdf).fit()
print(f"계수 확인(coeffients){lm.params},\np-value{lm.pvalues},\nR² : {lm.rsquared}")
print("상관계수(r)**2 == 결정계수",0.78222442 ** 2) # r² = R²

# ols모델 정보테이블 확인
# print(lm.summary())
# 필요한 테이블만 끊어서 볼 수 있다.
print(lm.summary().tables[0])
print('summary--------------')
print(lm.summary().tables[1])
print()

# 기존데이터로 예측하기 - 설명력 62%내에서 설명한다.
x_new = pd.DataFrame({'tv':advdf.tv[:3]})
print(x_new)
print('실제값 :', advdf.sales[:3].values)  # [22.1          10.4           9.3]
print("예측값 :",lm.predict(x_new).values) # [17.97077451  9.14797405  7.85022376]
print("직접계산 :",lm.params.tv * 230.1 + lm.params.Intercept)
print()

# 경험하지 않은 tv 광고비에 따른 상품판매량 예측
my_new = pd.DataFrame({'tv':[100, 350, 780]})
print("예측 상품 판매량:",lm.predict(my_new).values) # [11.78625759 23.6704177  44.11117309]
print()

# 시각화
# plt.scatter(advdf.tv, advdf.sales)
# plt.xlabel("tv 광고비")
# plt.ylabel('상품판매량')
# ypred = lm.predict(advdf.tv) # 추세선
# plt.plot(advdf.tv, ypred, c='r')
# plt.title('단순선형회귀')
# plt.grid(True)
# plt.show()
# print()

# 적절성 선행조건 확인하기=================================================================
print('-'*10,'단순선형회귀 모델이므로 적절성 선행조건중 잔차의 정규성, 선형성 확인','-'*10)

# 잔차(Residual) 구하기
from scipy.stats import shapiro # 정규성 확인
import statsmodels.api as sm

fitted = lm.predict(advdf) # 단순, 다중선형회귀 전부 사용가능, 내부에서 알아서 tv Column을 사용, 이방법을 더 추천
# ==lm.predict(advdf.tv) - 단순선형회귀에서만 사용 가능
residual = advdf['sales'] - fitted
print('실제값 :', advdf['sales'][:5].values)
print('예측값 :', fitted[:5].values)
print('잔차값 :', residual[:5].values)
print('잔차 평균값 :', np.mean(residual[:5])) # 1.67388299

# 잔차의 정규성 검정=====================================================================
print('-'*10,'잔차가 정규성을 따르는지 확인하기','-'*10)
# shapiro test, Q-Q plot사용
stat, p = shapiro(residual)
print(f'shapiro 통계량 : {stat:.4f}, p-value : {p:.4f}') 
# p-value : 0.2133 > α 이므로 정규성 만족
print("p-value 정규성 만족" if p > 0.05 else "p-value 정규성 위배")
print()

# Q-Q plot 시각화 - 정규성을 만족하고 있다.
# sm.qqplot(residual, line='s')
# plt.title("Q-Q Plot로 정규성 만족 확인")
# plt.show()
# 끝부분의 커브를 그리며 데이터값이 밖으로 나가고 있다.
# 이런 case는 좋지 않다. log취해서 변동성이 큰 데이터를 확인할 수 있다.

# 선형성 검정=============================================================================
print('-'*10,'잔차가 선형성을 따르는지 확인하기 : 독립변수의 변화에 종속변수도 변화하나 특정한 패턴이 있으면 안됨','-'*10)
from statsmodels.stats.diagnostic import linear_reset # 선형성 확인 모델
reset_result = linear_reset(lm, power=2, use_f=True) # 설명값을 제곱, f값을 이용할거야
print("reset_result 결과(p) : ", reset_result.pvalue)
print("선형성 만족" if reset_result.pvalue > 0.05 else "선형성 위배")
print()

# 선형성 시각화
# 직선 회귀모형이 맞다면 잔차에 규칙적인 모양이 남으면 안 되기 때문에 시각화 확인
# 각 잔차점을 하나씩 이은 선이 아니라, 잔차들의 전체적인 흐름 패턴을 부드럽게 요약한 선
# sns.regplot(x=fitted, y=residual, lowess=True, 
            # line_kws={'color':'magenta'},   # 선형성 시각화
            # scatter_kws={'color':'gray'},   # 잔차값 시각화 
            # )
# 기준선 그리기 - 잔차 그래프에서는 y=0이 중심선 역할을 한다
# plt.plot([fitted.min(), fitted.max()], [0,0], '--', color='blue')
# plt.xlabel('예측값(Fitted Values)')
# plt.ylabel('잔차(Residuals)')
# plt.title('선형성 확인용 잔차 그래프')
# plt.show()
# 패턴이 일정하게 가기 때문에 선형성을 만족하고 있다고 할 수 있다.
# 점 : 각 관측치의 잔차들
# 빨간 선이 -- 0선 주변에서 일정하게 움직여야 한다
# 선형성은 대체로 괜찮고, 등분산성은 추가 확인이 필요해 보이는 형태


# 등분산성(Homoscedasticity) 검정======================================================
print('-'*10,'등분산성(Homoscedasticity) 검정 : 모든 x값에서 오차의 퍼짐정도(분산)가 일정해야한다.','-'*10)
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(resid=residual, exog_het=sm.add_constant(advdf['tv']))
bp_stat, bp_pvalue = bp_test[0], bp_test[1]
print(f'het_breuschpagan test 통계량: {bp_stat}, p-value:{bp_pvalue}')
print()
print("등분산성 만족" if bp_pvalue > 0.05 else "등분산성 위배")
print()

'''참고 
Cook's distance
특정 데이터가 회귀모델에 얼마나 영향을 주는지 확인
영향력 있는 관측치(이상치)를 탐지하는 진단 방법
데이터가 적을때, 이상치가 의심스러울 때, 모델결과가 이상하게 나올 때
일반적으로 1값이 넘어가면 관측치를 영향점으로 판별
이상값은 cook’s distance(쿡의 거리)위에 존재한다
'''
print("-"*10,"Cook's distance","-"*10)
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lm).cooks_distance  # return값이 두개 = cooks_distance, index

# 쿡거리가 가장 큰 5개 확인
print(cd.sort_values(ascending=False).head())
print()

# 영향력이 큰 (쿡거리가 가장 큰) 관측치 원본 확인
print(advdf.iloc[[35, 178, 25, 175, 131]])
'''    tv  radio  newspaper  sales
35   290.7    4.1        8.5   12.8
178  276.7    2.3       23.7   11.8
25   262.9    3.5       19.5   12.0
175  276.9   48.9       41.8   27.0
131  265.2    2.9       43.0   12.7

-> 대부분 tv광고비는 매우 높으나 sales가 낮음 - 모델이 예측하기 어려운 포인트들이다.
'''
# 시각화
fig = sm.graphics.influence_plot(lm, alpha=0.05, criterion='cooks')
plt.show()

# 시각화 2-1 - stem
cooks_d = lm.get_influence().cooks_distance[0]
plt.figure(figsize=(10, 4))
# 각 관측치의 Cook's Distance를 세로선 형태로 표시
plt.stem(cooks_d, basefmt=" ")
plt.axhline(4 / len(cooks_d), linestyle='--')
plt.xlabel('관측치 인덱스')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")
plt.show()

# 시각화 2-2
cooks_d = lm.get_influence().cooks_distance[0]
threshold = 4 / len(cooks_d)
plt.figure(figsize=(10, 4))
plt.stem(cooks_d, basefmt=" ")
plt.axhline(threshold, linestyle='--')
# 기준값을 넘는 인덱스만 표시
for i, v in enumerate(cooks_d):
    if v > threshold:
        plt.text(i, v, str(i), ha='center', va='bottom', fontsize=9)
plt.xlabel('관측치 인덱스')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")
plt.show()

# 시각화3 - bar
cooks_d = lm.get_influence().cooks_distance[0]
plt.figure(figsize=(10, 4))
plt.bar(range(len(cooks_d)), cooks_d)
plt.axhline(4 / len(cooks_d), linestyle='--')
plt.xlabel('관측치 인덱스')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")
plt.show()

# 시각화 3-2
cooks_d = lm.get_influence().cooks_distance[0]
threshold = 4 / len(cooks_d)
plt.figure(figsize=(10, 4))
plt.bar(range(len(cooks_d)), cooks_d)
plt.axhline(threshold, linestyle='--')
# 기준값을 넘는 인덱스만 표시
for i, v in enumerate(cooks_d):
    if v > threshold:
        plt.text(i, v, str(i), ha='center', va='bottom', fontsize=9)
plt.xlabel('관측치 인덱스')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")
plt.show()
print()


print("="*30,' 다중선형회귀 ',"="*30)
# 다중선형회귀 모델 만들기 ols===========================================================
# x :tv,radio,newspaper  y:sales
lm_mul = smf.ols(formula='sales ~ tv + radio + newspaper', data=advdf).fit()
print(lm_mul.summary())
# 모델은 Prob (F-statistic):  1.58e-96 의미 O
# 모델의 설명계수 -> Adj. R-squared:  0.896 - 독립변수가 종속변수의 분산을 89.6% 설명한다.
# 독립변수 newspaper P>|t|  0.860 의미 X - 버리는게 좋다.