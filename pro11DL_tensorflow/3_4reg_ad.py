'''
다중선형회귀

    feature : tv. radio, newspaper
    target : sales
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv")
print(data.head(3))
del data['no']
print(data.head(2))

fdata = data[['tv','radio','newspaper']]
ldata = data.iloc[:, [-1]]
print(fdata.head(2))
print(ldata.head(2))

# feature 간 단위의 차이가 클 경우 정규화/표준화 작업이 모델성능에 도움을 준다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale

# 정규화
# scaler = MinMaxScaler(feature_range=(0, 1))
# fedata = scaler.fit_transform(fdata)
# print(fedata[:3])
fedata = minmax_scale(fdata, axis=0, copy=True) # 행기준, 원본자료 보존
print(fedata[:3])

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
                fedata, 
                ldata, 
                shuffle=True, 
                test_size=0.3, 
                random_state=123
                # stratify <- 분류에서 사용, 회귀 사용 X
    )
print(x_train[:2], x_train.shape)   # (140, 3)
print(x_test[:2], x_test.shape)     # (60, 3)
print()

# 전처리가 모두 끝난 경우 모델 설계 및 실행
model = Sequential()
model.add(Input(shape=(3, )))
model.add(Dense(units=16, activation='relu')) # 은닉층을 크게 주고나서 작게주는게 성능이 가장 좋다
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='linear')) # activation 생략 가능 - 기본이 linear
print(model.summary())

# pip install pydot
# https://cafe.daum.net/flowlife/SBU0/13 - path설정
# https://graphviz.org/download/
# keras모델 구조를 이미지 파일로 저장
tf.keras.utils.plot_model(model, 
                        to_file = 'aaa.png', 
                        show_shapes = True,             # 각 layer의 입력/출력 shape표시
                        show_layer_names = True,        # layer 이름 표시
                        show_dtype = True,              # 데이터 타입 표시
                        show_layer_activations = True,  # activation 함수 표시
                        dpi = 96                        # 이미지 해상도
)

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2,
                    validation_split = 0.2 # 학습중 검증용으로 20%를 학습 중 검증용 사용
                    )
ev_loss = model.evaluate(x_test, y_test, verbose=0)
print('ev_loss :', ev_loss)
print()

# history값 확인하기
# print('history :',history.history)
# validation split이 있는 경우 val_loss, val_mse확인가능 없으면 값도 없어
print('\nhistory val_loss:',history.history['val_loss'])
print('\nhistory val_mse:',history.history['val_mse'])

print('\nhistory loss:',history.history['loss'])
print('\nhistory mse:',history.history['mse'])
print()

# loss 시각화 하기
import matplotlib.pyplot as plt
import koreanize_matplotlib

plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['loss'], label='loss')
plt.legend()
plt.show()

# R2
from sklearn.metrics  import r2_score
print('설명력 :',r2_score(y_test, model.predict(x_test)))
# 설명력 : 0.6647094488143921
print()

# predict
pred = model.predict(x_test[:5])
print('예측값 :',pred.ravel())
print('실제값 :',y_test[:5].values.ravel())
# 예측값 : [12.601319  8.310078 13.596421  9.951106 12.468611]
# 실제값 : [11.4  8.8 14.7 10.1 14.6]
print()

# Functional api를 사용하는 방법
# 다중 입출력, 분기구조, 병합구조 등 복잡한 신경망 모델 작성시 효과적
print('='*20,'Functional API 사용','='*20)
from tensorflow.keras.models import Model

# 입력층 정의
inputs = Input(shape=(3, ), name='input_layer') # name은 텐서보드 등에 사용
#은닉층 정의
x = Dense(units=16, activation='relu', name='hidden_layer1')(inputs)
x = Dense(units=8, activation='relu', name='hidden_layer2')(x)
# 출력층 정의
outputs = Dense(units=1, activation='linear', name='output_layer')(x)

# 모델생성 - 입력과 출력을 연결
func_model = Model(inputs=inputs, outputs=outputs)
print(func_model.summary())

# ▼▼▼ 여기부터는 다른모델과 방법이 같다 ▼▼▼
func_model.compile(optimizer='adam', loss='mse', metrics=['mse'])

history = func_model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2,
                    validation_split = 0.2 # 학습중 검증용으로 20%를 학습 중 검증용 사용
                    )
func_ev_loss = func_model.evaluate(x_test, y_test, verbose=0)
print('func_ev_loss :', func_ev_loss)

# R2
print('설명력 :',r2_score(y_test, func_model.predict(x_test)))
# 설명력 : 0.8083456754684448
print()