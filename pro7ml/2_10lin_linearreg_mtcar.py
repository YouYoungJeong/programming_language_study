'''
LinearRegression 클래스를 사용해
    mtcar DataSet 사용
'''
from sklearn.linear_model import LinearRegression
import statsmodels.api
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error 
# mean_absolute_error 이상치가 있을때 사용

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3))
print("상관계수 :\n", mtcars.corr(method='pearson'))
print()

# hp가 mpg에 영향을 주는 인과관계
x = mtcars[['hp']].values # 2차원 반환
print(x[:5])
y = mtcars.mpg.values # 1차원 반환
print(y[:5])

# 모델생성
lmodel = LinearRegression().fit(x, y)
print('slope(w) :',lmodel.coef_)
print("intercept(b) :", lmodel.intercept_)
print()

# 시각화
# plt.scatter(x, y)
# plt.plot(x, lmodel.coef_*x + lmodel.intercept_, c='r')
# plt.show()

# mpg예측
pred = lmodel.predict(x)
print('y^ :',np.round(pred[:5],1)) # [22.6 22.6 23.8 22.6 18.2]
print('y  :',np.round(y[:5],3))    # [21.  21.  22.8 21.4 18.7]
print()


# 모델 성능지표
# 모델끼리 비교할때는(모델 성능) R²하나만 보고 모델 판단 하면 안되고 설명력만 봄 - 이상치에 민감, 독립변수가 많으면 증가
# R²와MSE 또는 R²,RMSE 를 확인해야한다
# MSE   : 모델 내부 비교, 계산은 편리함. - 단위가 제곱한값
# RMSE  : 보고 / 해석용, 해석이 용이함. - 단위가 원래 단위가 됨.
# 회귀 평가지표는 고정된 점수 범위(score)가 없다. - 데이터 스케일에 따라 다름 
# => 모델끼리 상대적인 비교를 한다.
print('MSE :', mean_squared_error(y, pred))             # 13.9898
print("RMSE :", np.sqrt(mean_squared_error(y, pred)))   # 3.7402
print("R² :", r2_score(y, pred))                        # 0.60243
print()

# 새로운 hp로 mpg예측
print('새로운 hp로 mpg예측----------------------------------------------')
new_hp = [[100], [110], [120], [130]] # 4행1열의 값으로 준다(2차원)
new_pred = lmodel.predict(new_hp)
print("예측결과 :",np.round(new_pred.flatten(), 2)) # [23.28 22.59 21.91 21.23]
print()
