'''
ex31
앙상블(Ensemble) 모델 XGboost 분류 알고리즘
    모듈 설치해야한다!
    #    pip install xgboost # Boosting 제일 기본적인 모델
    #    pip install lightgbm # XGboost보다 성능이 우수하나 자료가 적으면 과적합 우려가 있다.
    Boosting 알고리즘을 구현한 분류/예측 모델
    Boosting은 약한 분류기에 대해 샘플에 일부를 보안해가며 
    순차적으로 학습해 강한 분류기를 만듦 
    성능이 매우 좋기 때문에 Overfitting을 주의 해야한다.

brest_cancer dataset사용
    [0:악성(M) 1:양성(B)]
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from lightgbm import LGBMClassifier # XGboost보다 성능이 우수하나 자료가 적으면 과적합 우려가 있다.
import lightgbm as lgb

data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print(x[:3])
print(y[:3])
print(x.shape, y.shape) # (569, 30) (569,)
print('label분포 :', {name:(y==i).sum() for i, name in enumerate(data.target_names)})
# {'malignant'(악성) : 212, 'benign'(양성): 357}
# 분포의 차이가 좀 있다

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=12, stratify=y)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (455, 30) (114, 30) (455,) (114,)

# XGB 모델1 생성하기
xgb_clf = xgb.XGBClassifier(
    booster = 'gbtree', # 분류모델(tree):gbtree, 선형모델:gblinear
    max_depth=6, # 개별 결정 트리 최대 깊이
    n_estimators=200, # 약합 분류기의(의사결정나무) 갯수
    eval_metric = 'logloss', # 모델 평가
    random_state = 42
)
xgb_clf.fit(x_train, y_train)

# LGB 모델2 생성하기
lgb_clf = lgb.LGBMClassifier(
    n_estimator=200,random_state = 42, verbose=-1 # 로그숨기기
)
lgb_clf.fit(x_train, y_train)

# 예측 / 평가
pred_xgb = xgb_clf.predict(x_test)
pred_lgb = lgb_clf.predict(x_test)
print("XGBClassifier acc :", np.round(accuracy_score(y_test, pred_xgb), 5)) # 0.96491
print("LGBMClassifier acc :", np.round(accuracy_score(y_test, pred_lgb), 5)) # 0.99123
print()

# Feature 중요도 : gain값 기준으로 통일
booster = xgb_clf.get_booster()
xgb_gain = pd.Series(booster.get_score(importance_type='gain')) # Series로 값을 받아옴

lgb_gain = pd.Series(
    lgb_clf.booster_.feature_importance(importance_type='gain'),
    index = x_train.columns
)
# print(xgb_gain)
# print(lgb_gain)

# xgb_gain / xgb_gain.sum() : 각 feature의 기여도를 비율로 만들기
xgb_gain_pct = 100 * xgb_gain / (xgb_gain.sum() if xgb_gain.sum() != 0 else 1)
lgb_gain_pct = 100 * lgb_gain / (lgb_gain.sum() if lgb_gain.sum() != 0 else 1)

# 사용되지 않은 feature는 0으로 채움
xgb_gain_pct = xgb_gain_pct.reindex(x_train.columns).fillna(0)
lgb_gain_pct = lgb_gain_pct.reindex(x_train.columns).fillna(0)

comp_df = pd.DataFrame({
    'XGBoost (gain %)':xgb_gain_pct,
    'LightGBM (gain %)':lgb_gain_pct
}).sort_values('XGBoost (gain %)', ascending=False)

# 중요 Feature(변수) top10
print(comp_df.head(10))

# 시각화
topk = 5
top = comp_df.head(topk)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
xmax = float(np.ceil(top.max().max())) # 두 모델의 최대값

for ax, col in zip(axes, ['XGBoost (gain %)','LightGBM (gain %)']):
    ax.barh(top.index, top[col])
    ax.set_title(f'{col.split()[0]} Feature importance')
    ax.set_xlabel("Importance (%)")
    ax.set_xlim(0, xmax)
plt.tight_layout()
plt.show()