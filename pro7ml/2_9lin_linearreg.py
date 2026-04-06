'''
LinearRegression 클래스 사용하기
    (전통적인 방법, 딥러닝 선형회귀랑 비교할 줄 알아야한다.)
    평가 score 정리
'''
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.linear_model import LinearRegression # summary() 지원하지 X
# 결정계수, 설명된분산점수, 오차 확인을 위해 class 호출
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
# 정규화 class 호출
from sklearn.preprocessing import MinMaxScaler


# 데이터 생성
sample_size = 100
np.random.seed(1)
x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 30
print(x[:5])
print(y[:5])
print('상관 계수 :', np.corrcoef(x, y)[0, 1]) # 0.999393

# 독립변수 x를 정규화(Normalization) - 0 ~ 1사이의 범위내 자료로 변환
# sklearn은 모델을 생성할때 입력데이터(독립변수)를 2차원의 데이터를 원하기 때문에 차원을 바꿔서 사용한다.
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
print(x[:5])
print(x_scaled[:5])

# 시각화 하기
# plt.scatter(x_scaled, y)
# plt.show()

# 모델 생성
model = LinearRegression().fit(x_scaled, y)

print('model :', model)
print('회귀계수(slope) :', model.coef_)
print('회귀계수(intercept = bias) :', model.intercept_)
print('결정계수(R²) :', model.score(x_scaled, y))
y_pred = model.predict(x_scaled) # y^ 100개
print('y^ =',y_pred[:5])
print('y  =',y[:5]) 
print()

# 모델 성능 확인 함수 작성
def myRegScoreFunc(y_true, y_pred):
    # 결정계수(R²) : 
    #   실제 관측값의 분산대비 예측값의 분산을 계산하여 데이터 예측의 정확도 성능 측정 지표
    print(f"R²(결정계수) : {r2_score(y_true, y_pred)}")
    
    # 설명분산점수(Explained Variance Score) : 
    #   모델이 데이터의 분산을 얼마나 잘 설명하는지 나타내는 지표.
    #   머신러닝 회귀 모델이 실제 데이터의 분산(Variance, 흩어짐 정도)을 얼마나 잘 설명(예측)하는지를 측정하는 평가지표
    #   오차 분산이 작으면 점수가 높아진다. 1에 가까울수록 모델이 데이터의 경향성을 잘 설명하는 우수한 모델
    print(f"설명분산점수(Explained Variance Score) : {explained_variance_score(y_true, y_pred)}")

    # 평균 제곱 오차(Mean Squared Error, MSE) :
    #   오차**2/n - 오차를 제곱해 평균을 구함 - 오차가 커질수록 손실함수 값이 빠르게 중가함
    #   값이 작으면 모델 성능이 우수 (err는 작을 수록 좋다.)
    #   예측 모델의 정확도를 평가하기 위해 사용하는 지표
    print(f"평균 제곱 오차(Mean Squared Error, MSE) : {mean_squared_error(y_true, y_pred)}")

    # 평균 제곱근 오차(Root Mean Squared Error, RMSE) : 
    #   모델의 예측값과 실제 관측값 사이의 평균 차이를 측정하는 데 사용되는 표준 지표
    #   제곱 잔차(차이)의 평균의 제곱근
    #   값이 낮을수록 적합도가 높고 예측이 더 정확함 - (MAE)보다 이상치에 더 민감
    mse = mean_squared_error(y_true, y_pred)
    print(f"평균 제곱근 오차(Root Mean Squared Error, RMSE) : {np.sqrt(mse)}")
    # 9.281592052012815 (평균적으로 약 9만큼의 오차가 있는것이다.)


myRegScoreFunc(y, y_pred) # 함수에 y값과 y^값을 보냄
# cost = y - y^
# cost를 잔차제곱합의 최소화 하려면 추세선이 잘 그려져야함
print()

# 분산이 크게 다른 x, y값 사용
print('분산이 크게 다른 x, y값 사용---------------------------')
x2 = np.random.normal(0, 1, sample_size)
y2 = np.random.normal(0, 100, sample_size) + x * 30
print(x2[:5])
print(y2[:5])
print('상관 계수 :', np.corrcoef(x2, y2)[0, 1]) # -0.05484

# 정규화
scaler = MinMaxScaler()
x_scaled2 = scaler.fit_transform(x2.reshape(-1, 1))
print(x2[:5])
print(x_scaled2[:5])

# 모델 생성
model2 = LinearRegression().fit(x_scaled2, y2)

print('model2 :', model2)
print('회귀계수(slope) :', model2.coef_)
print('회귀계수(intercept = bias) :', model2.intercept_)
print('결정계수(R²) :', model2.score(x_scaled, y2)) #  -0.107490002
# => 분산이 너무 다른 데이터로 만든 모델은 의미 X