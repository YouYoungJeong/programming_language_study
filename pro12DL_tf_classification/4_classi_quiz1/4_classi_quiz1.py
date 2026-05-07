'''
문제2) 21세 이상의 피마 인디언 여성의 당뇨병 발병 여부에 대한 
dataset을 이용하여 당뇨 판정을 위한 분류 모델을 작성한다.
피마 인디언 당뇨병 데이터는 아래와 같이 구성되어 있다.

    Pregnancies: 임신 횟수
    Glucose: 포도당 부하 검사 수치
    BloodPressure: 혈압(mm Hg)
    SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
    Insulin: 혈청 인슐린(mu U/ml)
    BMI: 체질량지수(체중(kg)/키(m))^2
    DiabetesPedigreeFunction: 당뇨 내력 가중치 값
    Age: 나이
    Outcome: 5년 이내 당뇨병 발생여부 - 클래스 결정 값(0 또는 1)
    당뇨 판정 칼럼은 outcome 이다.   1 이면 당뇨 환자로 판정
train / test 분류 실시
모델 작성은 Sequential API, Function API 두 가지를 사용한다.
ModelCheckPoint, EarlyStopping 사용
loss, accuracy에 대한 시각화를 실시한다.
출력결과는 Web framework를 사용하시오.
https://github.com/pykwon/python/blob/master/testdata_utf8/pima-indians-diabetes.data.csv
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import os

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/pima-indians-diabetes.data.csv", header=None)
print(df.head())
print(df.info())
print(df.shape)     # (768, 9)
df = df.values

# feature / label
x = df[:,:-1]
print(x.shape)      # (768, 9)
y = df[:,-1]
print(set(y))   # [1 0]
# print(x[:2])
# print(y[:2])

# train test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, stratify=y, random_state=0
)

# Scailing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model 생성 1 - Sequential
print('='*30,' 방법1) Sequential API','='*30)
model1 = Sequential([
    Input(shape=(x_train.shape[1], )),
    Dense(units=16, activation='relu'),
    Dense(units=8, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# model compile - 학습 방법 설정
# 학습 전에 loss, optimizer, metric을 정해서 모델이 공부할 방법을 알려주는 단계
model1.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(learning_rate=0.001), 
    metrics=['accuracy']
)

# 조기종료
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 저장
MODEL_DIR = '../classif_quiz/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 모델 저장에 대한 조건설정
modelpath = MODEL_DIR + '/seq_model.keras'
chkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                        mode='auto', save_best_only=True)

# model fit
seq_history = model1.fit(x_train, y_train, epochs=1000,
                    validation_split=0.2, 
                    batch_size = 32,
                    callbacks = [early_stop, chkpoint],
                    verbose=0)

# 평가
seq_eval = model1.evaluate(x_test, y_test, verbose=0)
print(seq_eval) 
print(f'평가 결과 : 손실(loss)={seq_eval[0]:.4f},정확도={seq_eval[1]:.4f}')
print()

# 시각화(loss, acc)
plt.figure(figsize=(12, 5))
# loss
plt.subplot(1, 2, 1)
plt.plot(seq_history.history['loss'], label='loss', color='pink')
plt.plot(seq_history.history['val_loss'], label='val_loss', color='darkgreen', alpha=0.9)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('seq_history - Loss')
plt.legend()
# acc
plt.subplot(1, 2, 2)
plt.plot(seq_history.history['accuracy'], label='accuracy', color='pink')
plt.plot(seq_history.history['val_accuracy'], label='val_accuracy', color='darkgreen', alpha=0.9)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('seq_history - Accuracy')
plt.legend()
plt.show()

# model 생성 2 - Function API =====================================================
print('='*30,' 방법2) Functional API ','='*30)
inputs = Input(shape=(x_train.shape[1], ))
outputs = Dense(units=16, activation='relu')(inputs)
outputs = Dense(units=8, activation='relu')(outputs)
outputs = Dense(units=1, activation='sigmoid')(outputs)
model_func = Model(inputs=inputs, outputs=outputs)
print(model_func.summary())

# 모델 저장에 대한 조건설정
modelpath2 = MODEL_DIR + '/func_model.keras'
chkpoint = ModelCheckpoint(filepath=modelpath2, monitor='val_loss', 
                        mode='auto', save_best_only=True)

# model compile
model_func.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(learning_rate=0.001), 
    metrics=['accuracy']
)

# model fit
func_history = model_func.fit(x_train, y_train, epochs=1000,
                    validation_split=0.2, 
                    batch_size = 32,
                    callbacks = [early_stop, chkpoint],
                    verbose=0)

seq_eval2 = model_func.evaluate(x_test, y_test, verbose=0)
print(seq_eval2) 
print(f'평가 결과 : 손실(loss)={seq_eval2[0]:.4f},정확도={seq_eval2[1]:.4f}')
print()


# 시각화(loss, acc)
plt.figure(figsize=(12, 5))
# loss
plt.subplot(1, 2, 1)
plt.plot(func_history.history['loss'], label='loss', color='pink')
plt.plot(func_history.history['val_loss'], label='val_loss', color='darkgreen', alpha=0.9)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('func_history - Loss')
plt.legend()
# acc
plt.subplot(1, 2, 2)
plt.plot(func_history.history['accuracy'], label='accuracy', color='pink')
plt.plot(func_history.history['val_accuracy'], label='val_accuracy', color='darkgreen', alpha=0.9)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('func_history - Accuracy')
plt.legend()
plt.show()