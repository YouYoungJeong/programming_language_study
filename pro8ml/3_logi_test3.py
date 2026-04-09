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

# 모델의 성능 파악
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
#     P         N
# T [[141(TP)    1(FN)]
# F [ 6 (FP)    152(TN)]]
acc = (141 + 152) / 300       # (TP + TN) / 전체수
recall = 141 / (141 + 1)      # TP / (TP + FN)
precission = 141 / (141 + 6)  # TP / (TP + FP)
specificity = 152 / (6 + 152) # TN / (FP + TN) - 특이도
fallout = 6 / (6 + 152)      # FP / (FP + TN) - 위양성율(fpr)
print('acc : ', acc)   
print('recall(TPR - roc curve y축, 1에 근사하면 좋다.) : ', recall)   
print('precission : ', precission)   
print('specificity : ', specificity)   
print('fallout(FPRate - roc curve x축, 0에 근사하면 좋다.)', fallout)   
print('fallout(== 1-specificity)', 1-specificity)   
print()

acc_score = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
print("모델 정확도 :",acc_score)
print("모델 정밀도 :",precision)
print("모델 재현율 :",recall)
print()

cl_rep = metrics.classification_report(y_test, y_pred)
print('분류모델 리포트 :\n',cl_rep)
print()

# ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, model.decision_function(x_test))
print('fpr :',fpr)
print('tpr :',tpr)
print('AUC(Area Under the Curve : ROC Curve의 면적 : 1에 근사할 수록 좋다. )출력')
print('AUC :',metrics.auc(fpr, tpr))

# ROC Curve
plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0,1],[0,1], 'k--', label='landom classifier line(AUC:0.5)')
plt.plot([fallout],[recall],'ro', ms=6) # fpr, tpr 출력
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve')
plt.legend()
plt.show()

print("입력값 예측하기------------------------------------------------")
# ['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage']
dtime = float(input("현재 페이지에 보낸 시간을 입력하세요: "))
age = int(input("나이를 입력하세요: "))
income = float(input("소득수준를 입력하세요: "))
daily = float(input("평균 인터넷 사용 시간를 입력하세요: "))

if not isinstance(dtime, float) or dtime < 0 :
    print("페이지에 보낸 시간이 0보다 작거나 실수가 아닙니다.양의 실수 값을 입력해 주세요")
elif  not isinstance(age, int) or age < 0:
    print("나이가 0보다 작거나 정수가 아닙니다.양의 정수 값을 입력해 주세요")
elif not isinstance(income, float) or income < 0 :
    print("소득수준이 0보다 작거나 실수가 아닙니다.양의 실수 값을 입력해 주세요")
elif not isinstance(daily, float) or daily < 0 :
    print("평균 인터넷 사용 시간이 0보다 작거나 실수가 아닙니다.양의 실수 값을 입력해 주세요")
else:
    newdf = pd.DataFrame({'Daily Time Spent on Site':[dtime], 
                            'Age':[age],
                            'Area Income':[income],
                            'Daily Internet Usage':[daily]})
    sc.fit(newdf)
    new_data = sc.transform(newdf)
    
    new_pred = model.predict(new_data)
    if np.around(new_pred) == 1 :
        print('입력값 예측결과 :', new_pred)
        print("광고를 클릭 합니다.")
    else:
        print('입력값 예측결과 :', new_pred)
        print("광고를 클릭 안합니다.")
