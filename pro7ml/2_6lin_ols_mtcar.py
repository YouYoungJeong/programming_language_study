'''
단순선형회귀 - mtcars.csv사용
'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api 
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars)
print(mtcars.columns)
# ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
print(mtcars.info())
# x:hp(마력수), y=mpg(연비)

# 상관관계 확인하기 - 강한 음의 상관관계
print(np.corrcoef(mtcars.hp, mtcars.mpg)[0, 1]) # -0.7761683
print(np.corrcoef(mtcars.wt, mtcars.mpg)[0, 1]) # -0.8676593
print()

# 시각화 하기
# plt.scatter(mtcars.hp, mtcars.mpg)
# plt.xlabel('마력수')
# plt.ylabel('연비')
# plt.show()

# 단순선형회귀
print('단순선형회귀-----------------------------------------------------------------')
result = smf.ols(formula='mpg ~ hp', data=mtcars).fit()
print(result.summary())
# F-statistic:1.79e-07 < α
# R² = 0.602 
# y^ = -0.0682*x + (30.0989) + stderr(ε)
print('마력수 110에 대한 연비 예측값 : ',-0.0682 * 110 + (30.0989)) # 22.5969
# 사용해야 하는 방식은 result.predict
print('마력수 110에 대한 연비 예측값 : ',result.predict(pd.DataFrame({'hp':[110]}))) # 22.59375
print()

# 다중선형회귀
print('다중선형회귀-----------------------------------------------------------------')
result2 = smf.ols(formula='mpg ~ hp+wt', data=mtcars).fit()
print(result2.summary())
# Prob (F-statistic) : 9.11e-12
# Adj. R-squared : 0.815
# d :  37.2273, w1(hp):-0.0318, w2(wt):-3.8778
print('마력수 110 + 무게5에 대한 연비 예측값 : ',\
      (-0.0318 * 110) + (-3.8778*5) + (37.2273)) # 14.3403
# 사용해야 하는 방식은 result.predict
print('마력수 110에 대한 연비 예측값 : ',\
    result2.predict(pd.DataFrame({'hp':[110], 'wt':[5]}))) # 14.343092
print()

# 추정치 구하기
print('추정치 구하기- 차체무게를 입력해 연비 추정-------------------------------------')
result3= smf.ols(formula='mpg ~ wt', data=mtcars).fit()
print(result3.summary())
print('결정계수 : ', result3.rsquared)
pred = result3.predict() # 0.75283
print('result3 연비 예측값 :',pred[:5])
# [23.28261065 21.9197704  24.88595212 20.10265006 18.90014396]

# 새로운 차체 무게로 연비 추정 (df의 col으로 볼 수 있다.)
mtcars.wt = float(input('차체무게 입력:'))
new_pred = result3.predict(pd.DataFrame(mtcars.wt))
print(f'차체무게 {mtcars.wt[0]}일때 예상 연비는 {new_pred[0]}')