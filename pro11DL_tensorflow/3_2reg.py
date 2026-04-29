'''
단순 선형회귀 모델 작성
    tensorflow는 x는 2차원, y는 1~2차원 둘다 가능
    선형회귀 -> 은닉층 많이 안만들어도 되고, 마지막은 꼭 하나에 linear 사용
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import numpy as np

# fearture, label을 2차원 형태로 입력하기 위함
xdata = np.array([1, 2, 3, 4, 5], dtype='float32').reshape(-1, 1) # 2차원 변경
ydata = np.array([1.2, 2.0, 3.0, 3.5, 5.5]).reshape(-1, 1)
print('상관계수 :', np.corrcoef(xdata.ravel(), ydata.ravel())) # 0.97494708

# 모델 생성하기
model = Sequential()
model.add(Input((1,)))
model.add(Dense(units=5, activation='relu')) # 은닉층 - 선형회귀는 은닉층을 많이 만들 필요는 없다.
model.add(Dense(units=1, activation='linear')) # linear : 계산된 값을 그대로 출력하는 활성함수
print(model.summary())
# 선형회귀 -> 은닉층 많이 안만들어도 되고, 마지막은 꼭 하나에 linear 사용


model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
# loss='mse' : Mean Squared Error <- 회귀분석 모델에서 사용
# optimizer='sgb' : 경사하강법 <- 기본속성값밖에 못씀
# tensorflow.keras.optimizers.SGD는 학습률을 조정할 수 있다.

model.fit(x=xdata, y=ydata, epochs=30, batch_size=1, verbose=1, shuffle=True)
# shuffle=True : default
loss_eval = model.evaluate(x=xdata, y=ydata)
print('loss_eval :',loss_eval)

pred = model.predict(xdata)
print('pred :', pred.ravel())
print('real :', ydata.ravel())
# pred : [1.1154574 2.1375082 3.1595592 4.18161   5.2036605]
# real : [1.2 2.  3.  3.5 5.5]
print()

# 설명력
print('결정계수 : R2, 설명력')
from sklearn.metrics import r2_score
print('설명력 :', r2_score(ydata, pred)) # 설명력 : 0.925

# 시각화
import matplotlib.pyplot as plt
plt.scatter(xdata, ydata, color='green', marker='o', label='real')
plt.plot(xdata, pred, 'b--', label='pred')
plt.legend()
plt.show()

# 새로운 값으로 예측
new_x = np.array([1.5, 5.7, -3.0]).reshape(-1, 1)
new_pred = model.predict(new_x)
print('새값 예측 결과 :', new_pred.ravel())
# [ 1.7068818   6.382128   -0.27812743]