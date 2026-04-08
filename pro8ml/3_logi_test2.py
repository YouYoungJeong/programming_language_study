'''
[로지스틱 분류분석 문제2] 
게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
안경 : 값0(착용X), 값1(착용O)
예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X
'''
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split # 모델 샘플링 추출 모듈
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
import joblib  

df = pd.read_csv("bodycheck.csv")
print(df.info())
print(df.shape, df.isna().sum()) # (20, 6) , 0

# 데이터 추출
glasses_data = df.loc[:,['안경유무','TV시청','게임']]
# print(glasses_data.head(2))
x = glasses_data[['TV시청','게임']]
y = glasses_data['안경유무']
print()

# train - test -scale
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print('x_train :',x_train.shape,'\n', x_train[:3]) # (14, 2)
print('y_train :',y_train.shape,y_train[:3]) # (14,)
print('x_test :',x_test.shape,'\n', x_test[:3]) # (6, 2)
print('y_test :',y_test.shape, y_test[:3]) # (6,)
print()

# 분류모델 생성
print('분류 모델 생성-----------------------------------------------------')
model = LogisticRegression(C=0.06)
# 학습시키기
model.fit(x_train, y_train)

# 분류 예측
print('분류 예측---------------------------------------------------------')
y_pred = model.predict(x_test)
print("예측값 :", y_pred)           # [1 1 0 1 1 0]
print("실제값 :", y_test.values)    # [1 1 0 1 1 0]
print()

# 분류 정확도 확인 1
print("분류 정확도 확인 1 sklearn accuracy_score 사용 --------------------")
print(f'{accuracy_score(y_test, y_pred)}') 
print()

# 분류 정확도 확인 2
print("분류 정확도 확인 2 pandas crosstable사용 --------------------------")
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측치'], colnames=['관측치'])
print(con_mat)
print((con_mat[0][0]+con_mat[1][1]) / len(y_test)) 
print()

# 분류 정확도 확인 3
print("분류 정확도 확인 3 model.score함수 사용---------------------------")
print('test score : ', model.score(x_test, y_test))    
print('train score : ', model.score(x_train, y_train)) 
print()
#==================================================================================
# 학습 후 검증이 된 모델 저장 후 읽기
#==================================================================================
# 저장
joblib.dump(model, 'logimodel_test.pkl') # 확장자명은 sav, model... 이 3개가 일반적으로 많이 사용
del model
# 읽기
read_model = joblib.load('logimodel_test.pkl')

#==================================================================================
# 입력값 예측하기
#==================================================================================
print("입력값 예측하기------------------------------------------------")
tv = int(input("tv시청시간을 입력하세요: "))
game = int(input("게임시간을 입력하세요: "))

if not isinstance(tv, int) or tv < 0 :
    print("tv 시청 시간이 0보다 작거나 정수가 아닙니다.양의 정수 값을 입력해 주세요")
elif  not isinstance(game, int) or game < 0:
    print("게임 시간이 0보다 작거나 정수가 아닙니다.양의 정수 값을 입력해 주세요")
else:
    newdf = pd.DataFrame({'TV시청':[tv], '게임':[game]})
    new_pred = read_model.predict(newdf)
    if np.around(new_pred) == 1 :
        print('입력값 예측결과 :', new_pred)
        print("안경을 착용 합니다.")
    else:
        print('입력값 예측결과 :', new_pred)
        print("안경을 착용 안합니다.")
