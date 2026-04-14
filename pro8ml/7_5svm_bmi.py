'''ex37
SVM (Support Vector Machine) - BMI
    BMI(Body Mass Index)는 키와 몸무게로 체지방량을 추정하여 비만도를 간편하게 측정하는 지표
    공식 : 몸무게(kg) / 키(m)의 제곱
    ex) 키 170, 몸무게 68 => 68 /(170 / 100) **2
print(68 /(170 / 100) **2)
'''
'''
BMI 데이터 만들기

import random
random.seed(12)
def cald_bmiFunc(h, w):
    bmi = w / (h / 100) ** 2
    if bmi < 18.5 : return 'thin'
    if bmi < 25.0 : return 'normal'
    return 'fat'
print(cald_bmiFunc(170, 68))
print()

# csv 파일 생성
fp = open('bmi.csv', mode='w')
fp.write("height,weight,label\n") # 제목

# 무작위 데이터 생성
cnt = {'thin':0, 'normal':0, 'fat':0}

for i in range(50000):
    h = random.randint(150, 200)
    w = random.randint(35, 100)
    label = cald_bmiFunc(h, w)
    cnt[label] += 1
    fp.write('{0},{1},{2}\n'.format(h,w,label))

fp.close()
'''
# =======================================================================================
# bmi data를 SVM으로 분류
# =======================================================================================
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

df = pd.read_csv("bmi.csv")
print(df.head(2))
print(df.shape)         # (50000, 3)
print(df.info())        # int64(2), object(1)
print(df.describe())

label = df['label']
print(label[:2])

# df w, h는 단위가 달라서 정규화가 필요 (0~1사이의 값으로 맞춰주기)
w = df['weight'] / 100
print(w[:2].values)     # [0.69 0.79]
h = df['height'] / 200
print(h[:2].values)     # [0.9  0.96]

wh = pd.concat([w, h], axis=1)
print(wh.head(2))

# label은 dummy화 시키기 - 문자열을 숫자열로 바꿔주기
label = label.map({'thin':0, 'normal':1, 'fat':2})
print(label.head(3))

# train-test spilt
x_train, x_test, y_train, y_test = train_test_split(wh, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape) # (35000, 2) (15000, 2)

# model생성
model = svm.SVC(C=0.01, kernel='rbf') # 작을 수록 과적합 방지 세게 들어감

# 훈련
model.fit(x_train, y_train)
print(model) # SVC(C=0.01)

# 예측하기
spred = model.predict(x_test)
print('예측값 : ', spred[:10])          # [2 0 1 1 0 0 2 1 0 0]
print('실제값 : ', y_test[:10].values)  # [2 0 1 1 0 0 2 1 0 0]

# 정확도
sc_score = metrics.accuracy_score(y_test, spred)
print("분류 정확도 :", sc_score)        # 0.97053

# 교차 검증 모델 생성하기
# from sklearn import model_selection
# cross_val = model_selection.cross_val_score(model, wh, label, cv=3)
# print('3회 각 정확도 :',cross_val)      # [0.96940061 0.96586068 0.96681867]
# print("평균 정확도 :",cross_val.mean()) # 0.9673599

# 새로운 값을 예측
new_data = pd.DataFrame({'weight':[66, 88], 'height':[188, 160]})
new_data['weight'] = new_data['weight'] / 100
new_data['height'] = new_data['height'] / 200
new_pred = model.predict(new_data)
print('새로운 값 예측 결과 :',new_pred)

# 시각화
df2 = pd.read_csv('bmi.csv', index_col=2)
def scatterFunc(label, color):
    b= df2.loc[label]
    plt.scatter(b['weight'], b['height'], c=color, label=label)

scatterFunc("fat", 'magenta')
scatterFunc("normal", 'orange')
scatterFunc("thin", 'green')
plt.legend()
plt.show()