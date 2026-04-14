'''
Santander Customer Satisfaction - train.csv dataset 사용
    (https://www.kaggle.com/datasets/ioramishvili/santaner/data)
    Santander 은행의 고객 만족 여부 분류
    클래스(label)명은 target이고 0:만족, 1:불만족

Feature (특징/피처): 모델의 입력값(x), Label (라벨/타겟): 모델의 출력값(y)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
# pd.set_option('display.max_columns', None)

# 데이터 불러와서 DataFrame에 넣기(정제 되지 않은 data는 이게 중요하다.)
df = pd.read_csv("snatander_customer_satisfaction/train_san.csv", encoding='latin-1')
# print(df.head(2)) # 'TARGET' <= class,label
print(df.shape)     # (76020, 371)
# print(df.info())    # float64(111), int64(260)

# 전체 데이터에서 만족과 불만족의 비율 확인하기
print(df['TARGET'].value_counts())  # 0(만족):73012 , 1(불만족):3008
unsatisfied_cnt = df[df['TARGET']==1].TARGET.count()
total_cnt = df.TARGET.count()
print(f'불만족 비율 : {unsatisfied_cnt / total_cnt * 100 :.5f}%') # 3.95685%

# feature의 분포 확인
print(df.describe()) # [8 rows x 371 columns]

# 이상치 처리
# var3 min = -999999.000000 <- 이상치 의심(중앙값으로 바꿈)
df['var3'].replace(-999999.000000, 2, inplace=True)
df.drop("ID", axis=1, inplace=True) # ID는 식별자
print(df.describe()) # [8 rows x 370 columns]

# Feature와 Label 분리 작업
x_features  = df.iloc[:, :-1]
y_label = df.iloc[:,-1]
print('x_features shape : ',x_features.shape) # (76020, 369)

# train - test - split
x_train, x_test, y_train, y_test = train_test_split(
                    x_features, y_label, test_size=0.2, random_state=0)
train_cnt = y_train.count()
test_cnt = y_test.count()
print(x_train.shape, x_test.shape) # (60816, 369) (15204, 369)
print(f"학습 데이터 레이블 값 분포 비율 : {y_train.value_counts() / train_cnt}")
# 0 : 0.960964 , 1 : 0.039036
print(f"검증 데이터 레이블 값 분포 비율 : {y_test.value_counts() / test_cnt}")
# 0 : 0.9583 , 1 : 0.0417
print()

# 모델 생성
xgb_clf = XGBClassifier(n_esimators=5, # n_esimators원래는 500~1000개 이상은 줘야함
                        random_state=12, 
                        eval_metric='auc'
                    ) # eval_metric='auc' 평가지표 : 분류모델이기 때문에  AUC (회귀모델은 MES, SMES...)



# 모델 훈련(fit)
xgb_clf.fit(x_train, y_train,
            eval_set=[(x_train, y_train),(x_test, y_test)]) 

# 모델 성능 평가(ROC/AUC Score)
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(x_test)[:, 1]) # [:, 1] 결과는 Vector처리를 위해 줌.
print(f"xgb_roc_score : {xgb_roc_score:.5f}" ) # xgb_roc_score : 0.82683
print()

# 예측하기
pred = xgb_clf.predict(x_test)
print('예측값 :', pred[:5])
print('실제값 :', y_test[:5].values)
print()

# 정확도 확인하기
from sklearn import metrics
print("분류 정확도 :", metrics.accuracy_score(y_test, pred)) # 0.95797
print()

# 최적의 파라미터 알아보기
xgb_clf = XGBClassifier(n_esimators=5)

params = {"max_depth":[5, 7],           # tree 깊이
        'min_child_weight':[1, 3],      # 관측치에 대한 가중치합 최소
        'colsample_bytree':[0.5, 0.75]  # feature 비율
        }
gridcv = GridSearchCV(xgb_clf, param_grid=params)
gridcv.fit(x_train, y_train, eval_set=[(x_train, y_train)])
print("CV의 최적 Parameter :", gridcv.best_params_)
# {'colsample_bytree': 0.5, 'max_depth': 5, 'min_child_weight': 3}
xgb_roc_score = roc_auc_score(y_test, gridcv.predict_proba(x_test)[:, 1], average='macro')
'''
average='macro'
다중 클래스 분류 모델의 성능을 평가하는 방법
매크로 평균(Macro-average)
    각 클래스의 지표(Precision, Recall, F1)를 각각 계산해 산술 평균하는 방식
마이크로 평균(Micro-average)
    전체 클래스의 True Positive/False Positive/False Negative를 합산하여 한 번에 계산하는 방식
'''
print(f"xgb_roc_score : {xgb_roc_score:.5f}" ) # xgb_roc_score : 0.83566
print()

# 최적의 파라미터로 모델 생성
xgb_clf2 = XGBClassifier(n_esimators=5, random_state=12,
                        max_depth=5, min_child_weight=3, colsample_bytree=0.5)
xgb_clf2.fit(x_train, y_train,
            eval_set=[(x_train, y_train)])

# 예측하기
pred2 = xgb_clf2.predict(x_test)
print('예측값 :', pred2[:5])
print('실제값 :', y_test[:5].values)
print()

# 모델 성능 평가(ROC/AUC Score)
xgb_roc_score2 = roc_auc_score(y_test, xgb_clf2.predict_proba(x_test)[:,1], average='macro') 
print(f"xgb_roc_score : {xgb_roc_score2:.5f}" ) # xgb_roc_score : 0.83578
print()

# 정확도 확인하기
from sklearn import metrics
print("분류 정확도 :", metrics.accuracy_score(y_test, pred2)) # 0.958103
print()

# 중요 feature 시각화
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
plot_importance(xgb_clf2, ax=ax, max_num_features=20)
plt.show()