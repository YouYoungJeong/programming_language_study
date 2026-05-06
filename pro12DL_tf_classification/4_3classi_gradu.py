'''tf16
https://www.kaggle.com/datasets/jyotikumarrout/graduation
    binary.csv 데이터를 이용하여 미국 대학원 입학여부를 분류하는 모델을 작성하시오.

    label(class) : admit
    feature : gre, gpa, rank
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import os

df = pd.read_csv('binary.csv')
print(df.head(3))
print(df.info())

# 전처리 : rank는 연속형이 아니라 범주형 자료이므로 원핫 처리
df = pd.get_dummies(df, columns=['rank'], dtype=int)
print(df.head())
print(df.info())
print()

# feature, label분리하기
x = df.drop('admit', axis=1)
y = df['admit']
print(x.head(3))
print(y.head(3))

# scaling 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x) # 굳이 먼저 할 필요는 없어 성능이 안좋을때 시행

# train / test spilt
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42, stratify=y )

# model 생성
# print(x_train.shape[1]) # 6
model = Sequential([
    Input(shape=(x_train.shape[1], )),
    Dense(units=16, activation='relu'),
    Dense(units=8, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# model compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

# model.fit
history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test), # train으로 학습하는 중간에 test로 검증하는것도 가능
                    epochs=100, batch_size=32, verbose=2
)

# 검정 하기
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'test결과 - 손실:{loss:.4f}, 정확도:{acc:.4f}')

# 시각화(loss, acc)
plt.figure(figsize=(12, 5))
# loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='loss', color='pink')
plt.plot(history.history['val_loss'], label='val_loss', color='darkgreen', alpha=0.9)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.legend()
# acc
plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], label='acc', color='pink')
plt.plot(history.history['val_acc'], label='val_acc', color='darkgreen', alpha=0.9)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Accuracy')
plt.legend()
plt.show()

# 사용자가 입력한 데이터 예측
gre = float(input('gre 점수 입력'))
gpa = float(input('gpa(학점) 점수 입력'))
rank = int(input('rank 입력: (1 ~ 4)'))
# 입력된 rank 원핫인코딩 처리
rank_encoded = [0, 0, 0, 0]
rank_encoded[rank - 1] = 1

user_input = np.array([[gre, gpa] + rank_encoded])
print('user_input :', user_input) #  [[730.    3.5   0.    1.    0.    0. ]]

user_scaled = scaler.transform(user_input)
new_pred = model.predict(user_scaled)
prob = new_pred[0][0]
print('합격 확률 : ', prob)
if prob >= 0.5:
    print('합격 가능성이 높다.')
else:
    print('불합격 가능성이 높다.')
