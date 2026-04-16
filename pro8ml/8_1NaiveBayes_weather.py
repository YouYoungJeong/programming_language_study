'''
Naive Bayes Algorithm을 이용한 분류
    weather.csv
'''
import pandas as pd
import numpy as np

# 데이터 확인하기
df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv")
print(df.head(2))
print(df.info())
print(set(df['RainToday']), set(df['RainTomorrow'])) # {'No', 'Yes'} {'No', 'Yes'}
print()

# ============================= 데이터 전처리 작업 ============================= 
df = df.drop("Date", axis=1)

# 범주형 처리하기
df['RainToday'] = df['RainToday'].map({'Yes':1, "No":0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1, "No":0})
print(df.head(2))
print()

# 결측치 처리하기 - 평균처리
df['Sunshine'] = df['Sunshine'].fillna(df['Sunshine'].mean()) 

# Feature, label 나누기
x = df.drop('RainTomorrow', axis=1) # Features
y = df['RainTomorrow']              # lable, class

# ============================= train test spilt ============================= 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

# ============================= Naive Bayes 모델학습 ============================= 
from sklearn.naive_bayes import GaussianNB # 연속형 데이터이므로 가우시안 
# BernoulliNB - 이진데이터일때 사용\
# MultinomialNB 텍스트일때 사용
model = GaussianNB()
model.fit(x_train, y_train)

# ============================= 예측 및 평가 ============================= 
from sklearn.metrics import accuracy_score, confusion_matrix
npred = model.predict(x_test)
print('분류 정확도 :', accuracy_score(y_test, npred)) # 0.87837
print('confusion_matrix(혼돈 행렬):\n', confusion_matrix(y_test, npred))
#  [[55  6]
#  [ 3 10]]
print()

# ============================= 교차 검증 ============================= 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5)
print("교차 검증 결과 에서 각 fold :",scores,"\n평균 :",scores.mean())
# 교차 검증 결과 에서 각 fold : [0.72972973 0.82191781 0.79452055 0.8630137  0.83561644] 
# 평균 : 0.8089596445760829
print()

# ============================= Feature 중요도 분석 =============================
# RandomForest처럼 코드가 있는게 아니라 직접 구해줘야함.
# 평균GaussianNB모듈에서 지원하는 멤버 : 각클래스별 feature 평균을 구해준다.
# Feature가 정규분포를 따른다는 가정하에 클래스별  
mean_0 = model.theta_[0] # RainTomerrow가 0인 경우(비 안오는 날 평균)
mean_1 = model.theta_[1] # RainTomerrow가 1인 경우(비 오는날 평균)
# 각 feature가 '비오는날 vs 비 안오는 날에 얼마나 차이가 나는가'에 대한 값
importance = np.abs(mean_1 - mean_0) 
feat_impo = pd.DataFrame({
    'feature' : x.columns,
    'importance' : importance
    }).sort_values(by='importance', ascending=False)
print("Feature 중요도")
print(feat_impo)
print()

# ============================= importance에 대한 시각화 =============================
import matplotlib.pyplot as plt
import koreanize_matplotlib

plt.figure()
plt.bar(feat_impo['feature'], feat_impo['importance'])
plt.xlabel("Feature")
plt.ylabel("Feature 중요도(평균 차이)")
plt.xticks(rotation=45) # 글씨 방향 설정
plt.tight_layout()
plt.show()

# ============================= 새로운 자료로 예측 =============================
new_data = pd.DataFrame([{
    'MinTemp'  :12.3,
    'MaxTemp'  :27.0,
    'Rainfall' :0.0,
    'Sunshine' :10.0,
    'WindSpeed':8.0,
    'Humidity' :40,
    'Pressure' :1005.0,
    'Cloud'    :1,
    'Temp'     :20.0,
    'RainToday':0
}])
newpred = model.predict(new_data)
print("예측 결과 :","비 옴" if newpred == 1 else '비 안옴')
print("확률은 ", model.predict_proba(new_data))