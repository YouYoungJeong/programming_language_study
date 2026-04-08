'''
[로지스틱 분류분석 문제3]
Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
얘를 사용해도 됨   'testdata/advertisement.csv' 
참여 칼럼 : 
    - Daily Time Spent on Site : 사이트 이용 시간 (분)
    - Age : 나이,
    - Area Income : 지역 소득,
    - Daily Internet Usage :일별 인터넷 사용량(분),
    - Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )

광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
데이터 간 단위가 큰 경우 표준화 작업을 시도한다.
모델 성능 출력 : 정확도, 정밀도, 재현율, ROC 커브와 AUC 출력
새로운 데이터로 분류 작업을 진행해 본다.
'''
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split # 모델 샘플링 추출 모듈
from sklearn.preprocessing import StandardScaler     # 표준화
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import joblib  

# 데이터 확인하기
df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/advertisement.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
# ['Daily Time Spent on Site', 'Age', 'Area Income',
#    'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country',
#    'Timestamp', 'Clicked on Ad']

# 데이터 나누기
x = df[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage']]
y = df['Clicked on Ad']
print(x[:3])
print(y[:3])
print()

# 조건 1. 데이터 간 단위가 큰 경우 표준화 작업을 시도한다.
# sns.boxplot([x['Daily Time Spent on Site'],x['Age'],x['Daily Internet Usage']])
# plt.show()
# sns.boxplot(x['Area Income'])
# plt.show()
# 데이터 크기 범위 확인 - Area Income만 단위가 높기 때문에 스케일 진행

# train - test - scale
print("train_test_spilt (7 : 3)-------------------------------------------")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print('x_train :',x_train.shape,'\n', x_train[:3])  # (700, 4)
print('y_train :',y_train.shape,y_train[:3])        # (700,)
print('x_test :',x_test.shape,'\n', x_test[:3])     # (300, 4)
print('y_test :',y_test.shape, y_test[:3])          # (300,)
print()

# Scaling - 독립변수(feature)만 표준화 진행, 종립변수는 범주형(0,1)인데 표준화 왜해~
print("Scaling------------------------------------------------------------")
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3] ,'\n', x_test[:3])
print()


# 분류모델 생성
print('분류 모델 생성 예측하기--------------------------------------------')
model = LogisticRegression()

# 학습시키기
model.fit(x_train, y_train)

# 분류 예측
y_pred = model.predict(x_test)
print("예측값 :", y_pred[:5])           
print("실제값 :", y_test[:5].values)    
print()

# Roc curve의 판별경계선 설정용 결정함수 사용
print(' Roc curve--------------------------------------------')
f_value = model.decision_function(x_test) # 평가는 학습에 쓰지 않은 x_test로 해야함
print('f_value : ',f_value[:10])
print()