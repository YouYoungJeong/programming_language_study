'''
모델 생성방법 (3가지)
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import optimizers
import numpy as np
from sklearn.metrics import r2_score

# 데이터 생성: 공부시간에 따른 성적 결과 예측
xdata = np.array([1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)
ydata = np.array([15, 32, 39, 55, 60], dtype=np.float32).reshape(-1, 1)

# =======================================================
# 모델 생성방법1
# Sequential API 사용 - 매우 기초
# =======================================================
print('='*20,'Sequential API 사용','='*20)
model = Sequential()    # 계층구조를 만듦
model.add(Input((1,)))  # tuple로 줘야함
model.add(Dense(units=4 , activation='relu')) # 은닉층 노드 4개
model.add(Dense(units=1 , activation='linear')) # 출력층 값이 하나니까 노드 1개

# model = Sequential([
#             Input((1,)),
#             Dense(units=4 , activation='relu'),
#             Dense(units=1 , activation='linear')
# ])
print(model.summary())
opti = optimizers.SGD(learning_rate=0.001)
model.compile(loss='mse', optimizer=opti, metrics=['mse'])
history = model.fit(x=xdata, y=ydata, batch_size=1, epochs=100, verbose=0)
loss_metrics=model.evaluate(x=xdata, y=ydata)
ypred = model.predict(xdata, verbose=0)

print("loss_metrics :", loss_metrics)
print('설명력 :', r2_score(ydata, ypred))
print('실제값 :', ydata.ravel())
print('예측값 :', ypred.ravel())

# 시각화
import matplotlib.pyplot as plt
plt.scatter(xdata, ydata, color='green', marker='o', label='real')
plt.plot(xdata, ypred, 'b--', label='pred')
plt.legend()
plt.show()

# MSE 변화량 시각화
plt.plot (history.history['mse'], label='mse')
plt.xlabel('epochs')
plt.legend()
plt.show()
print()

# =======================================================
# 모델 생성방법2
# Functional API 사용 - 매우 중요!(실무)
# 유연한 구조를 가짐
#   - 입력 자료로 여러층을 공유하거나 다양한 종류의 입출력 모델 생성 가능
#   - 데이터 흐름이 비순차적인 경우에도 효과적
#   - 생성가능 모델 : 다중 입력값 모델, 다중 출력값 모델, 공유층 활용 모델
#   - 다중 입출력, 분기구조, 병합구조 등 복잡한 신경망 모델 작성시 효과적
# =======================================================
print('='*20,'Functional API 사용','='*20)
from tensorflow.keras.models import Model
inputs = Input(shape=(1,)) # 입력 크기 지정
# 무조건!! 이전층을 다음층 함수에 입력으로 사용하기 위해 할당 해야함
output1 = Dense(units=4, activation='relu')(inputs)     # 은닉층
outputs = Dense(units=1, activation='linear')(output1)  # 출력층

model2 = Model(inputs, outputs)

opti2 = optimizers.SGD(learning_rate=0.001)
model2.compile(loss='mse', optimizer=opti2, metrics=['mse'])
history2 = model2.fit(x=xdata, y=ydata, batch_size=1, epochs=100, verbose=0)
loss_metrics2 = model2.evaluate(x=xdata, y=ydata)
ypred2 = model2.predict(xdata, verbose=0)

print("loss_metrics2 :", loss_metrics2)
print('설명력 :', r2_score(ydata, ypred2))
print('실제값 :', ydata.ravel())
print('예측값 :', ypred2.ravel())
print()

# =======================================================
# 모델 생성방법3-1
# Sub classing 사용
#   - Model을 상속받아 직접 모델 생성
#   - 아주 하이테크한 논문에 사용됨
# =======================================================
print('='*20,'Sub classing 사용','='*20)
# MyModel은 Model을 상속받았기 때문에 Keras 모델처럼 사용할 수 있다.
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__() # 부모클래스 생성자에게 클래스와 생성자를 보내
        self.d1 = Dense(units=4, activation='relu')
        self.d2 = Dense(units=1, activation='linear')

    # x : input매개변수
    # (부모 클래스의)메소드 오버라이딩 : Model class 안에 있는 함수를 다시 호출
    # call, build 은 자동 호출
    def call(self, x): # Input class에 객체를 생성하지 않고 call 메소드의 input매개변수 이용
        x = self.d1(x)
        return self.d2(x)
    
model3 = MyModel()

opti3 = optimizers.SGD(learning_rate=0.001)
model3.compile(loss='mse', optimizer=opti3, metrics=['mse'])
# fit  할때 call이 호출
history3 = model3.fit(x=xdata, y=ydata, batch_size=1, epochs=100, verbose=0)
loss_metrics3 = model3.evaluate(x=xdata, y=ydata)
ypred3 = model3.predict(xdata, verbose=0)

print("loss_metrics3 :", loss_metrics3)
print('설명력 :', r2_score(ydata, ypred3))
print('실제값 :', ydata.ravel())
print('예측값 :', ypred3.ravel())
print()

# =======================================================
# 모델 생성방법3-2
# Sub classing - Custom Layer층 사용
# 
# =======================================================
print('='*20,'Sub classing - Custom Layer층 사용','='*20)
from tensorflow.keras.layers import Layer

class MyLayer(Layer):
    def __init__(self, units=1, **kwargs):
        super(MyLayer, self).__init__( **kwargs)
        self.units = units

    def build(self, input_shape): # w b값 정의, 내부적으로 call()호출
        print(f'build.input_shape = {input_shape}')
        
        # 입력데이터갯수 결정하지 못할때는 [-1]
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal', # 시작값 random
                                trainable=True)
        self.b = self.add_weight(shape=(self.units, ), 
                                initializer='zeros', # 초기치 0
                                trainable=True) 
    def call(self, inputs):
        # matmul : 행렬 곱
        return tf.matmul(inputs, self.w) + self.b #  y = wx + b

class MLP(Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.linear1 = MyLayer(2) # units=2
        self.linear2 = MyLayer(1) # units=1
    
    def call(self, inputs):
        net = self.linear1(inputs)
        net = tf.nn.relu(net)       #activation
        return self.linear2(net)

model4 = MLP()

opti4 = optimizers.SGD(learning_rate=0.001)
model4.compile(loss='mse', optimizer=opti4, metrics=['mse'])
# fit  할때 call이 호출
history4 = model4.fit(x=xdata, y=ydata, batch_size=1, epochs=100, verbose=0)
loss_metrics4 = model4.evaluate(x=xdata, y=ydata)
ypred4 = model4.predict(xdata, verbose=0)

print("loss_metrics4 :", loss_metrics4)
print('설명력 :', r2_score(ydata, ypred4))
print('실제값 :', ydata.ravel())
print('예측값 :', ypred4.ravel())
print()