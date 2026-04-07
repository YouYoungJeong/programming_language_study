'''
비선형회귀분석(Non-linear regression)
    직선의 회귀선을 곡선으로 변환해 보다 더 정확하게 
    데이터 변화를 예측하는 데 목적이 있다.

전통적인 방법
    선형가정이 어긋날 때(비정규성) 대처할 수 있는 방법으로 
    다항식항을 추가한 다항회귀모델 사용 - 다중X, 다항O
    입력데이터의 성질을 유지하면서 특징변환으로 선형 모델을 개선

    비선형 모델을 사용하기전에 이상치확인 후 제거하는것이 좋다.
    뚜렷하게 비선형모델인 경우에는 비선형모델을 사용하는게 좋다.
방법 : 
    다항식 특징을 추가하는 방법. 여러 방법 중 가장 일반적인 방법(PolynomialFeatures)
        y = w₀x₀ + w₁x₁+ w₂x₂ + w₃x₃ + ... + b
    log변환 - 곡선이 직선이 될 수 있다
    corve_fit
    의사결정트리사용 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])

# plt.scatter(x, y)
# plt.show()

print(np.corrcoef(x, y)[0, 1]) # 0.480761 - 우상향상관관계, 비선형모델

# 선형회귀 모델 적용해보기
from sklearn.linear_model import LinearRegression
# x1 = x.reshape(-1, 1) # 2차원 배열로 변경
x1 = x[:, np.newaxis]   # 2차원 배열로 변경
model = LinearRegression().fit(x1, y)
y_pred = model.predict(x1)
print('y^ :',y_pred)    # [2.  2.7 3.4 4.1 4.8]
print('y :',y)          # [4 2 1 3 7]
print('Residuals :',y-y_pred) # [ 2.  -0.7 -2.4 -1.1  2.2]
print('R² :', r2_score(y, y_pred)) # 0.2311

# plt.scatter(x, y)
# plt.plot(x, y_pred, c='r')
# plt.show()

# 비선형모델 작성
# 여러 방법 중 가장 일반적인 방법을 사용(PolynomialFeatures)
# y = w₀x₀ + w₁x₁+ w₂x₂ + w₃x₃ + ... + b
from sklearn.preprocessing import PolynomialFeatures # 다항식 특징을 추가

poly = PolynomialFeatures(degree=2, include_bias=False) # degree 열수, include_bias : 편향bias
x2 = poly.fit_transform(x1) # 특징 행렬을 만듦 : 열(data, Feature) 추가
print(x1)   # 5 * 1
print(x2)   # 5 * 2
# 모델 생성 - 특징 행렬값으로 학습
model2 = LinearRegression().fit(x2, y)
y_pred2 = model2.predict(x2)
print(f'y^(Poly) : y_pred2')    # [4.14285714 1.62857143 1.25714286 3.02857143 6.94285714]
print('y :',y)                  # [4 2 1 3 7]
print('Residuals :',y-y_pred2) # [-0.14285714  0.37142857 -0.25714286 -0.02857143  0.05714286]
print('R² :', r2_score(y, y_pred2)) # 0.98921
# 시각화
plt.scatter(x, y)
plt.plot(x, y_pred2, c='r')
plt.show()

