'''
문제2)
    https://github.com/pykwon/python/tree/master/data
    자전거 공유 시스템 분석용 데이터 train.csv를 이용하여 
    대여횟수에 영향을 주는 변수들을 골라(중요 feature찾기) 다중선형회귀분석 모델을 작성하시오.
    모델 학습시에 발생하는 loss를 시각화하고 설명력을 출력하시오.
    새로운 데이터를 input 함수를 사용해 키보드로 입력하여 
    대여횟수 예측결과를 콘솔로 출력하시오.
    casual	비회원 대여량	비회원 사용자의 자전거 대여 수
    registered	회원 대여량	등록 회원의 자전거 대여 수

# Tensorflow는 x = 2차원, y = 2차원 데이터를 받아서 학습을함
    pandas df로 파일을 받아 추출하면 데이터 추출시 다시 numpy로 바꿔주면 됨.
        데이터 추출
        x_data = df.drop('Close')
        y_data = df[['Close']]
        
        numpy로 형변환
        x_data = x_data.values.astype('float32')
        y_data = y_data.values.astype('float32')
        or
        x_data = x_data.to_numpy().astype('float32')
        y_data = y_data.to_numpy().astype('float32')
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
# feature 중요도 확인하기
from sklearn.ensemble import RandomForestRegressor
# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import optimizers

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv")
print(df.head())
print(df.shape) # (10886, 12)
print(df.info())
print(df.isnull().sum())

# datetime 변환
df["datetime"] = pd.to_datetime(df["datetime"])

# datetime 기반 파생변수 생성
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

# 원본 datetime 제거
df = df.drop("datetime", axis=1)
df = df.drop("registered", axis=1)
df = df.drop("casual", axis=1)

print(df.shape) # (10886, 14)
print(df.info())

# feature
x = df.drop(columns='count')
print(x.head(2))
print(x.shape) # (10886, 13)

# label
y = df['count']
y_data = df[['count']]
print(y[:2],y.shape) # (10886,) : 1차원
print(y_data[:2],y_data.shape) # (10886, 1) # 2차원


# Feature 중요도 확인하기
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=0
)
rf.fit(x, y)
importance_df = pd.DataFrame({
    'feature': x.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df) # ['hour','year','temp','workingday','month','dayofweek','atemp','humidity','weather','day']
# ['hour','workingday']
#         feature  importance
# 11        hour    0.606555
# 8         year    0.087732
# 4         temp    0.072040
# 2   workingday    0.048362
# 9        month    0.047741
# 12   dayofweek    0.040329
# 5        atemp    0.029106
# 6     humidity    0.027215
# 3      weather    0.014155
# 10         day    0.010058

# 중요도 높은 순서로 정렬
importance_df = importance_df.sort_values('importance', ascending=False)
# 그래프 크기 설정
plt.figure(figsize=(8, 5))
# 막대그래프 생성
plt.bar(
    importance_df['feature'],
    importance_df['importance']
)
plt.title('Feature Importance - 중요도 확인하기')
plt.xlabel('Feature')
plt.ylabel('Importance')
# x축 글자가 겹치지 않게 회전
plt.xticks(rotation=45)
# 그래프 여백 자동 조정
plt.tight_layout()
# 그래프 출력
plt.show()


# 중요 Feature추출 후 Tensorflow전용 데이터 만들기 - numpy형태로
x = x[['hour','year','temp','workingday','month','dayofweek','atemp','humidity','weather','day']]

# 스케일
scale = MinMaxScaler(feature_range=(0, 1))
x_data = scale.fit_transform(x)

# Tensorflow 데이터 형태 만들기
y_data = y_data.to_numpy().astype('float32')
print(x_data.shape, x_data.dtype, y_data.shape, y_data.dtype) # 

# train / test
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=123)
# print(x_train.shape, x_test.shape)  # (7620, 2) (3266, 2)

# model생성
model = Sequential()
model.add(Input(shape=(10,)))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=200, verbose=0, validation_split=0.1)

print(f'evaluate result : {model.evaluate(x_train, y_train, verbose=0)}')

# 결정계수 확인하기
pred = model.predict(x_test)
print(f'설명력 : {r2_score(y_test, pred)}') # 설명력 : 0.6374567151069641
print()

# history값 확인
# print('history loss : ', history.history['loss'])
# print('history val loss : ', history.history['val_loss'])
2
# loss 시각화
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()
plt.show()

# 새로운 데이터로 예측하기

# 1. 현재 시간 가져오기
from datetime import datetime
now = datetime.now()

# 2. 현재 시간에서 datetime 파생변수 생성
new_year = now.year
new_month = now.month
new_day = now.day
new_hour = now.hour
new_dayofweek = now.weekday()  # 월=0, 화=1, ..., 일=6

# 3. 사용자가 입력할 값
new_temp = float(input("temp(실제 온도) 입력: "))
new_workingday = int(input("workingday 입력 0/1: "))
new_atemp = float(input("atemp(체감 온도) 입력: "))
new_humidity = float(input("humidity(습도) 입력: "))
new_weather = int(input("weather 입력 1~4: "))

# 데이터 훈련순서
feature_cols = [
    'hour', 'year', 'temp', 'workingday', 'month',
    'dayofweek', 'atemp', 'humidity', 'weather', 'day'
]

# 새로운값 df로 만들기
new_x = pd.DataFrame({
    'hour': [new_hour],
    'year': [new_year],
    'temp': [new_temp],
    'workingday': [new_workingday],
    'month': [new_month],
    'dayofweek': [new_dayofweek],
    'atemp': [new_atemp],
    'humidity': [new_humidity],
    'weather': [new_weather],
    'day': [new_day]
})

#  형변환
new_x_data = new_x.to_numpy().astype('float32')
# scale
new_x_data = scale.transform(new_x_data)
# 예측하기
new_pred = model.predict(new_x_data)
print("예측 대여 횟수 :", int(new_pred[0][0]),'대')