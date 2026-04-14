'''
[SVM 분류 문제] 심장병 환자 데이터를 사용하여 분류 정확도 분석 연습
https://www.kaggle.com/zhaoyingzhu/heartcsv
https://github.com/pykwon/python/tree/master/testdata_utf8         Heartcsv

Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 
각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
dataset에 대해 학습을 위한 train과 test로 구분하고 
분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
임의의 값을 넣어 분류 결과를 확인하시오.     
정확도가 예상보다 적게 나올 수 있음에 실망하지 말자. ㅎㅎ
feature 칼럼 : 문자 데이터 칼럼은 제외
label 칼럼 : AHD(중증 심장질환)

데이터 예)
"","Age","Sex","ChestPain","RestBP","Chol","Fbs","RestECG","MaxHR","ExAng","Oldpeak","Slope","Ca","Thal","AHD"
"1",63,1,"typical",145,233,1,2,150,0,2.3,3,0,"fixed","No"
"2",67,1,"asymptomatic",160,286,0,2,108,1,1.5,2,3,"normal","Yes"
...
'''
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection

data = pd.read_csv("Heart.csv")
data = data.drop(columns=['Unnamed: 0'])
print(data.head(2))
print(data.info())
print(data.shape)   # (303, 14)
print(data.describe()) # Ca 결측치 처리하기
print('----------------------------------'*3)

# Ca NaN값 확인
print(data['Ca'].describe())
data['Ca'].fillna(data['Ca'].mean(), inplace=True)
print(data.isnull().sum())
print()

# object col 제거하기
data.drop(columns=['ChestPain','Thal'], inplace=True) 
print(data.shape)  # (303, 12)

# dataset label, feature구분하기
# feature - 표준화가 필요
features = data.iloc[:, :-1]
print(features.describe())
print(features.head(3))
print(features.isnull().sum())
print()

# label - AHD [0:No, 1:Yes]
label = data['AHD']
label = label.map({"No":0, "Yes":1})
print(label.head())
print(label.value_counts()) # 0 : 164 , 1 : 139
print()

# train-test-split
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3)
print(x_train.shape, x_test.shape) # (212, 11) (91, 11)
print()

# StandardScaler
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3] ,'\n', x_test[:3])
print()

# 모델 생성하기
model = svm.SVC(probability=True, random_state=10) # probability= predict_proba를 사용하기위한 옵션
model.fit(x_train, y_train)

# 예측하기
spred = model.predict(x_test)
sproba = model.predict_proba(x_test)
print('예측값 : ', spred[:10])          # [1 1 1 0 1 1 1 0 1 0]
print('실제값 : ', y_test[:10].values)  # [1 1 1 0 1 0 0 0 1 0]
print("확률 값 :\n",sproba[:5])
#  [[0.06804029 0.93195971]
#  [0.04269377 0.95730623]
#  [0.92702803 0.07297197]
#  [0.95381888 0.04618112]
#  [0.91980502 0.08019498]]
print()

# 성능 확인하기
# 정확도
sc_score = metrics.accuracy_score(y_test, spred)
print("분류 정확도 :", sc_score)        # 0.74725
print()

# 교차 검증 모델 생성하기
cross_val = model_selection.cross_val_score(model, x_train, y_train, cv=3)
print('3회 각 정확도 :',cross_val)      # [0.8028169  0.77464789 0.75714286]
print("평균 정확도 :",cross_val.mean()) # 0.778202
print()

# 새로운 값 예측하기
new_df = pd.DataFrame({
    'Age':      [45, 58, 63, 52, 67],
    'Sex':      [1, 0, 1, 1, 0],
    'RestBP':   [126, 138, 150, 120, 144],
    'Chol':     [210, 245, 289, 230, 310],
    'Fbs':      [0, 0, 1, 0, 1],
    'RestECG':  [0, 1, 2, 1, 2],
    'MaxHR':    [172, 148, 132, 160, 118],
    'ExAng':    [0, 1, 1, 0, 1],
    'Oldpeak':  [0.2, 1.4, 2.8, 0.6, 3.1],
    'Slope':    [1, 2, 2, 1, 3],
    'Ca':       [0, 1, 2, 0, 3]
})
sc = StandardScaler().fit(new_df)
new_df = sc.transform(new_df)
new_pred = model.predict(new_df)
print(new_pred) # [0 0 1 0 1]