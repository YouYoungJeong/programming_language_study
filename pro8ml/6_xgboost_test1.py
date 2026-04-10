'''
[XGBoost 문제] - softmax
kaggle.com이 제공하는 'glass datasets'          testdata 폴더 : glass.csv
유리 식별 데이터베이스로 여러 가지 특징들에 의해 7 가지의 label(Type)로 분리된다.
RI	Na	Mg	Al	Si	K	Ca	Ba	Fe	Type
                ...
glass.csv 파일을 읽어 분류 작업을 수행하시오.

glass의 type(target) = {1, 2, 3, 5, 6, 7} : 다중분류

현재 Type(target data)에 LabelEncoder 를 사용해하는 이유
    XGBoost 분류기는 타깃 라벨을 0부터 시작하는 정수 라벨로 받는 형태가 자연스러움

예측확률 평가 지표
    binary logloss / logloss
        주로 이진분류에서 사용
        0과 1 두 클래스의 확률 예측 품질 평가
    mlogloss
        주로 다중분류에서 사용
        클래스가 3개 이상일 때 각 클래스에 대한 예측 확률 분포를 평가
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 데이터 갯수 확인
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from  sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from lightgbm import LGBMClassifier # XGboost보다 성능이 우수하나 자료가 적으면 과적합 우려가 있다.
import lightgbm as lgb

glass = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/glass.csv")
print(glass.head(2))
print(glass.info()) # -> nan X, feature=float, target=int
print(set(glass.Type))              # {1, 2, 3, 5, 6, 7} -> 인코딩 필요(labelencoding)
print(Counter(glass.Type).values()) # [70, 76, 17, 13, 9, 29]
# 비율의 차이가 큼

print()
# x, y 데이터 추출
x = glass.drop('Type', axis=1)
y = glass['Type']
print(type(y)) # <class 'pandas.core.series.Series'>

# target data LabelEncoding
encoder = LabelEncoder() 
y = encoder.fit_transform(y)
print(set(y))  # 0, 1, 2, 3, 4, 5

# train- test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, stratify=y
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (149, 9) (65, 9) (149,) (65,)

# model 생성하기
# XGB 모델1 생성하기
xgb_clf = xgb.XGBClassifier(
    booster = 'gbtree', 
    # objective에 다중분류를 직접 적어주면 num_class는 무조건 넣어줘야함
    # objective로 직접 언급하지 않으면 num_class를 안써도 된다. 
    # 공식문서에 objective는 다중분류일때 넣는걸 추천
    objective='multi:softmax', # 다중분류 + 최종 클래스 번호 직접 출력
    num_class=6,
    max_depth=6, 
    n_estimators=200, 
    eval_metric = 'mlogloss'
)
xgb_clf.fit(x_train, y_train)

# LGB 모델2 생성하기
lgb_clf = lgb.LGBMClassifier(
    n_estimator=200, verbose=-1 # 로그숨기기
)
lgb_clf.fit(x_train, y_train)

# 예측 / 평가
pred_xgb = xgb_clf.predict(x_test)
pred_lgb = lgb_clf.predict(x_test)
print("XGBClassifier acc :", np.round(accuracy_score(y_test, pred_xgb), 5)) 
print("LGBMClassifier acc :", np.round(accuracy_score(y_test, pred_lgb), 5)) 
print()


# 새로운 데이터값으로 결과 확인하기
new_df = pd.DataFrame({
    'RI': [1.5172, 1.5215, 1.5160],
    'Na': [13.50, 13.10, 14.20],
    'Mg': [3.55, 3.40, 0.00],
    'Al': [1.30, 1.10, 2.10],
    'Si': [72.80, 71.90, 73.50],
    'K': [0.50, 0.10, 0.20],
    'Ca': [8.40, 9.80, 8.90],
    'Ba': [0.00, 0.00, 1.20],
    'Fe': [0.00, 0.05, 0.10]
})
pred_xgb2 = xgb_clf.predict(new_df)
pred_lgb2 = lgb_clf.predict(new_df)
print(pred_xgb2) # [0 0 5]
print(pred_lgb2) # [0 0 5]