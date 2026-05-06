'''tf14
딥러닝으로 이진분류 - 전통적인 방식인 LogisiticRegression의 확장
    
모델 생성 방법 4가지
    방법1) Sequential API 버전 - 빠른구현
    방법2) Functional API - 실무에서 주로 사용
    방법3) Functional API 버전2 - 다중 입력
    방법4) Model Subclassing 방식(완전 자유로운 형태 - 프로그램 능력이 중요)
    
    1번은 기본적으로 알아야하고 2번을 많이 사용해야한다.
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

x_data = np.array([[1,2],[2,3],[3,4],[4,3],[3,2],[2,1]], dtype=np.float32)
y_data = np.array([[0],[0],[0],[1],[1],[1]], dtype=np.float32)

#===============================================================================
# 방법1) Sequential API 버전 - 빠른구현
# 층을 순서대로 쌓는 단순 구조, 분기 구조나 다중 입출력 불가능
# 이진 분류(0또는1)는 loss='binary_crossentropy'(BCE) 예측값과 실제값 사이의 오차를 계산하는 손실 함수
# Dense 마지막은 무조건 : activation='sigmoid'
# 손실을 최소화 하기 위해 경사하강법을 사용
# z = w∙x + b -> sigmoid(z) -> BCE계산 -> 역전파 gradient(y^ - y) -> w와 b를 갱신
# ∙ : Matrix Multiplication(행렬 곱)
# 행렬 곱(Matrix Multiplication)은 앞 행렬의 열 개수와 뒤 행렬의 행 개수가 같을 때만 정의되며, 
# 앞 행렬의 '행'과 뒤 행렬의 '열'을 내적(곱의 합)하여 계산
#===============================================================================
print('='*30,' 방법1) Sequential API 버전 - 빠른구현','='*30)
# 1-1)
# model = Sequential([
#     Input(shape=(2,)),
#     Dense(units=1, activation='sigmoid')
# ])
# 1-2)
model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(units=4, activation='relu')) # 분류는 은닉층이 있으면 좋다. - 은닉층은 Relu가 좋다.
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())
print()

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model.fit(x_data, y_data, epochs=20, batch_size=1, verbose=0)
m_eval = model.evaluate(x_data, y_data, verbose=0)
print(m_eval) # [0.07844781130552292, 1.0]
print(f'평가 결과 : 손실(loss)={m_eval[0]:.4f},정확도={m_eval[1]:.4f}')

# 예측값, 실제값으로 시각화 하기(s곡선 형태)
import matplotlib.pyplot as plt
# 2차원 입력(x1, x2)을 가진 모델을 1차원 처럼 만들어 sigmoid(s곡선)를 보기위한 준비
x1_range= np.linspace(0, 6, 100)
x2_fixed = 2.5

# 입력 데이터 생성 : 두 배열을 합쳐서(stack) (x1, x2) 쌍 만들기
# np.full_like(x1_range, x2_fixed) : x1_range와 같은 길이의 배열 생성
x_vis = np.column_stack([x1_range, np.full_like(x1_range, x2_fixed)])
# print(x_vis)
# [[0.         2.5       ]
#  [0.06060606 2.5       ]
# x1변화에 따른 출력 확률값
y_prob = model.predict(x_vis, verbose=0)
x1_real = x_data[:, 0]
y_real = y_data.ravel()

plt.figure(figsize=(7, 5))
plt.plot(x1_range, y_prob, color='darkgreen', label='sigmoid curve')
plt.scatter(x1_real, y_real, color='magenta', label='True data')
plt.xlabel('x data')
plt.ylabel('probability')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 정확도 확인
from sklearn.metrics import accuracy_score
pred = model.predict(x_data, verbose=0)
pred_class = (pred >= 0.5).astype(int)
accuracy = accuracy_score(y_data, pred_class)
print(f'1) 정확도 : {accuracy:.4f}')

# 새로운 값 분류 예측
new_data = np.array([[1,2], [10,5]], dtype=np.float32)
pred = model.predict(new_data, verbose=0)
print(f'예측 확률1) : {pred.ravel()}') # 예측 확률 : [0.14681524 0.99999934] = 0 1

print('확률값 출력하는 방법 3가지')
print(f'예측결과 : {(pred >= 0.5).astype(int).ravel()}')
print(f'예측결과 : {[1 if i >= 0.5 else 0 for i in pred]}')
print(f'예측결과 : {np.where(pred >= 0.5 , 1, 0).ravel()}')
print()

#===============================================================================
# 방법2) Functional API - 실무에서 주로 사용
# 다중입출력 가능, 구조가 유연, 복잡한 모델에 효과적
#===============================================================================
print('='*30,' 방법2) Functional API - 실무에서 주로 사용','='*30)
from tensorflow.keras.models import Model

inputs = Input(shape=(2, ))
outputs = Dense(units=4, activation='relu')(inputs)
outputs = Dense(units=1, activation='sigmoid')(outputs)
model_func = Model(inputs=inputs, outputs=outputs)
print(model_func.summary())

model_func.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model_func.fit(x_data, y_data, epochs=20, batch_size=1, verbose=0)
m_eval2 = model_func.evaluate(x_data, y_data, verbose=0)
print(m_eval2)  # [0.4421076774597168, 1.0]
print(f'평가 결과2) : 손실(loss)={m_eval2[0]:.4f},정확도={m_eval2[1]:.4f}')
# 평가 결과 : 손실(loss)=0.4421,정확도=1.0000  - 결과가 똑같이는 안나옴 내부적으로 구조가 조금씩 다름

#===============================================================================
# 방법3) Functional API 버전2 - 다중 입력
# 다중입출력 가능, 구조가 유연, 복잡한 모델에 효과적
# 이전 : [x1, x2]-> Dense -> Dense -> 출력
# 다중 입력 : 입력을 따로 받아서 각각 특징을 뽑아 합치는 방식. 각각 따로 전처리가 가능함.
#            x1 -> Dense
#                       -> concat -> Dense -> 출력
#            x2 -> Dense
#===============================================================================
print('='*30,' 방법3) Functional API 버전2 - 다중 입력','='*30)
from tensorflow.keras.layers import Concatenate
# 입력 분리 
input1 = Input(shape=(1, ))
input2 = Input(shape=(1, ))
# 각각 처리
x1 = Dense(units=2, activation='relu')(input1)
x2 = Dense(units=4, activation='relu')(input2)

merged = Concatenate()([x1, x2])
output = Dense(units=1, activation='sigmoid')(merged) # 출력층
multi_model = Model(inputs=[input1, input2], outputs=[output])
print(multi_model.summary())

multi_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# 데이터를 분리해서 입력 - input1, input2
x1_data = x_data[:, 0].reshape(-1, 1)
x2_data = x_data[:, 1].reshape(-1, 1)
# fit x_data 분리한값 넣어주기
multi_model.fit([x1_data, x2_data], y_data, epochs=20, batch_size=1, verbose=0)
m_eval3 = multi_model.evaluate([x1_data, x2_data], y_data, verbose=0)
print(m_eval3)  # [0.6479906439781189, 0.6666666865348816]
print(f'평가 결과3) : 손실(loss)={m_eval3[0]:.4f},정확도={m_eval3[1]:.4f}')

#===============================================================================
# 방법4) Model Subclassing 방식(완전 자유로운 형태 - 프로그램 능력이 중요)
#===============================================================================
print('='*30,' 방법4) Model Subclassing 방식(완전 자유로운 형태 - 프로그램 능력이 중요)','='*30)

class MyModel(Model):
    def __init__(self): # layer 정의
        super(MyModel, self).__init__()
        self.dense1 = Dense(units=4, activation='relu')
        self.dense2 = Dense(units=1, activation='sigmoid')
    
    def call(self, inputs): # forward 연산 정의
        x = self.dense1(inputs)
        return self.dense2(x)
    
sub_model = MyModel() # 모델생성
sub_model.build(input_shape=(None, 2))
print(sub_model.summary())

sub_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
sub_model.fit(x_data, y_data, epochs=20, batch_size=1, verbose=0)
m_eval4 = sub_model.evaluate(x_data, y_data, verbose=0)
print(m_eval4) # [0.1452949494123459, 1.0]
print(f'평가 결과4) : 손실(loss)={m_eval4[0]:.4f},정확도={m_eval4[1]:.4f}')