"""
자전거 공유 시스템 분석용
: kaggle 사이트의 Bike Sharing in Washington D.C. Dataset를 편의상 조금 변경한 dataset을 사용함
columns : 
'datetime', 
'season'(사계절:1,2,3,4), 
'holiday'(공휴일(1)과 평일(0)), 
'workingday'(근무일(1)과 비근무일(0)), 
'weather'(4종류:Clear(1), Mist(2), Snow or Rain(3), Heavy Rain(4)), 
'temp'(섭씨온도), 'atemp'(체감온도), 
'humidity'(습도), 'windspeed'(풍속), 
'casual'(비회원 대여량), 'registered'(회원 대여량), 
'count'(총대여량)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from scipy import stats
pd.set_option('display.width', None) # 짤린 데이터 없이 다 보이게

plt.style.use('ggplot') # R에서 강력한 차트 모듈인 ggplot 스타일을 사용
train = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv",
                    parse_dates=['datetime'])
                        # datetime : object으로 안 받고 싷으면, parse_dates=['datetime']

print(train.info())     # datetime datetime64[ns] 으로 받아옴
print(train.dtypes)
print(train.shape) # (10886, 12)
print(train.columns)
'''['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
    'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']'''
print(train.head(3))
print(train.temp.describe())

# null 여부 파악
print(train.isnull().sum())

# EDA : 탐색적 분석 하기
# 데이터 가공
# 2011-01-01 00:00:00 년월일, 시분초 별도 칼럼 생성하기
# pandas의 datetime 전용 접근자dt 연산자 datetime64 타입일 때만 사용할 수 있음
train['year'] = train['datetime'].dt.year # dt 연산자 사용.
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second
print(train.head(1))
print(train.columns)

# 대여량 시각화
figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4) # oop 인터페이스 방식
figure.set_size_inches(15, 5) # 크기변경
sns.barplot(data=train, x='year', y='count', ax=ax1) #연도별 대여 횟수
sns.barplot(data=train, x='month', y='count', ax=ax2)
sns.barplot(data=train, x='day', y='count', ax=ax3)
sns.barplot(data=train, x='hour', y='count', ax=ax4)
ax1.set(ylabel='대여 수', title='연도별 대여')
ax2.set(ylabel='월', title='월별 대여')
ax3.set(ylabel='일', title='일별 대여')
ax4.set(ylabel='시간', title='시간별 대여')
plt.show()

# boxplot
# 계절별 시간별
fig, axes = plt.subplots(nrows=2, ncols=2) # 2행 2열 총 4개의 차트
figure.set_size_inches(12, 10)
sns.boxplot(data=train, y='count', orient='v', ax=axes[0][0])
sns.boxplot(data=train, y='count', x='season' ,orient='v', ax=axes[0][1])
sns.boxplot(data=train, y='count', x='hour' ,orient='v', ax=axes[1][0])
sns.boxplot(data=train, y='count', x='workingday' ,orient='v', ax=axes[1][1])
axes[0][0].set(ylabel='대여수', title='대여')
axes[0][1].set(xlabel='계절', ylabel='대여수',title='계절별 대여')
axes[1][0].set(xlabel='시간', ylabel='대여수',title='시간별 대여')
axes[1][1].set(xlabel='근무일', ylabel='대여수',title='근무일에 따른 대여')
plt.show()

#산점도 regplot() - 온도, 습도, 풍속 에 따른 대여량.
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
figure.set_size_inches(12, 5)
sns.regplot(x='temp', y='count', data=train, ax=ax1) # 온도
sns.regplot(x='humidity', y='count', data=train, ax=ax2) # 습도
sns.regplot(x='windspeed', y='count', data=train, ax=ax3) # 풍속

plt.show()



