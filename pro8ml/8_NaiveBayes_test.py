'''
[GaussanNB 문제] 
독버섯(poisonous)인지 식용버섯(edible)인지 분류
https://www.kaggle.com/datasets/uciml/mushroom-classification

feature는 중요변수를 찾아 선택, label:class
참고 : from xgboost import plot_importance

데이터 변수 설명 : 총 23개 변수가 사용됨.
여기서 종속변수(반응변수)는 class 이고 나머지 22개는 모두 입력변수(설명변수, 예측변수, 독립변수).
변수명 변수 설명
class      edible = e, poisonous = p
cap-shape    bell = b, conical = c, convex = x, flat = f, knobbed = k, sunken = s
cap-surface  fibrous = f, grooves = g, scaly = y, smooth = s
cap-color     brown = n, buff = b, cinnamon = c, gray = g, green = r, pink = p, purple = u, red = e, white = w, yellow = y
bruises        bruises = t, no = f
odor            almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s
gill-attachment attached = a, descending = d, free = f, notched = n
gill-spacing close = c, crowded = w, distant = d
gill-size       broad = b, narrow = n
gill-color      black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e, white = w, yellow = y
stalk-shape  enlarging = e, tapering = t
stalk-root    bulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing = ?
stalk-surface-above-ring fibrous = f, scaly = y, silky = k, smooth = s
stalk-surface-below-ring fibrous = f, scaly = y, silky = k, smooth = s
stalk-color-above-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y
stalk-color-below-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o,pink = p, red = e, white = w, yellow = y
veil-type      partial = p, universal = u
veil-color     brown = n, orange = o, white = w, yellow = y
ring-number none = n, one = o, two = t
ring-type     cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z
spore-print-color black = k, brown = n, buff = b, chocolate = h, green = r, orange =o, purple = u, white = w, yellow = y
population abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y
habitat       grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
from  sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import koreanize_matplotlib


#====================================================================================
# 데이터 전처리
#====================================================================================
mush = pd.read_csv("mushrooms.csv")
# print(mush.head(2))
# print(mush.shape)       # (8124, 23)
# print(mush.info())      # 결측치 X 전부, object

# lable encoding하기
encoder = LabelEncoder()
for col in mush.columns:
    mush[col] = encoder.fit_transform(mush[col])
# print(mush.head())
#e(식용) → 0 p(독) → 1

# 중요하지 않은 Feature 제거하기-['veil-color','veil-type','gill-attachment']
mush = mush.drop(columns=['veil-color','veil-type','gill-attachment'])
# print(mush.columns)

# Featuer, label 나누기
x = mush.drop('class', axis=1)
y = mush['class']
# print(x.head(2)) # [22 -> 19 columns]
# print(y[:2])

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
# print(x_train.shape, x_test.shape) # (6499, 19) (1625, 19)

#====================================================================================
#   중요 Feature 판단하기.
#====================================================================================

# XGB 모델 생성
xgb_clf = xgb.XGBClassifier(
    booster = 'gbtree', # 분류모델(tree):gbtree, 선형모델:gblinear
    max_depth=6, # 개별 결정 트리 최대 깊이
    n_estimators=200 # 약합 분류기의(의사결정나무) 갯수
)
xgb_clf.fit(x_train, y_train)

# Feature 중요도 : gain값 기준으로 통일
booster = xgb_clf.get_booster()
xgb_gain = pd.Series(booster.get_score(importance_type='gain')) # Series로 값을 받아옴

# xgb_gain / xgb_gain.sum() : 각 feature의 기여도를 비율로 만들기
xgb_gain_pct = 100 * xgb_gain / (xgb_gain.sum() if xgb_gain.sum() != 0 else 1)

# 사용되지 않은 feature는 0으로 채움
xgb_gain_pct = xgb_gain_pct.reindex(x_train.columns).fillna(0)

comp_df = pd.DataFrame({
    'XGBoost (gain %)':xgb_gain_pct
}).sort_values('XGBoost (gain %)', ascending=False)

# print(comp_df)

# 중요도 그래프
plt.figure(figsize=(10, 8))
plot_importance(
    xgb_clf,
    importance_type='gain',
    max_num_features=20,
    height=0.5
)
plt.title("XGBoost Feature Importance (gain)")
plt.show()

print()
# 사용되지 않은 Feature는 제거하고 사용하겠다.
# ['veil-color','veil-type','gill-attachment']

#====================================================================================
#   Naive Bayes 모델 생성후 학습 하기
#====================================================================================
model = GaussianNB()
model.fit(x_train, y_train)

#====================================================================================
#   Naive Bayes 모델 예측 및 평가
#====================================================================================
npred = model.predict(x_test)
print('분류 정확도 :', accuracy_score(y_test, npred)) # 0.87837
print('confusion_matrix(혼돈 행렬):\n', confusion_matrix(y_test, npred)) # 0.90215
#  [[798  44]
#  [115 668]]
print()

#====================================================================================
#   Naive Bayes 모델 교차 검증
#====================================================================================
scores = cross_val_score(model, x, y, cv=5)
print("교차 검증 결과 에서 각 fold :",scores,"\n평균 :",scores.mean())
# 교차 검증 결과 에서 각 fold : [0.63384615 0.97846154 0.80553846 0.93969231 0.55972906] 
# 평균 :  0.78345350
print()

#====================================================================================
#   Feature 중요도 분석
#====================================================================================
# Feature가 정규분포를 따른다는 가정하에 클래스별  
mean_0 = model.theta_[0] # 식용버섯
mean_1 = model.theta_[1] # 독버섯
# 각 feature가 '식용버섯인지 vs 독버섯인지에 얼마나 차이가 나는가'에 대한 값
importance = np.abs(mean_1 - mean_0) 
feat_impo = pd.DataFrame({
    'feature' : x.columns,
    'importance' : importance
    }).sort_values(by='importance', ascending=False)
# print("Feature 중요도")
# print(feat_impo)
print()

#====================================================================================
#   importance에 대한 시각화
#====================================================================================

plt.figure()
plt.barh(feat_impo['feature'], feat_impo['importance'])
plt.xlabel("Feature")
plt.ylabel("Feature 중요도(평균 차이)")
plt.xticks(rotation=80) # 글씨 방향 설정
plt.tight_layout()
plt.show()


#====================================================================================
#   새로운 값 예측하기
#====================================================================================
new_data = pd.DataFrame([
    {'cap-shape': 5, 'cap-surface': 2, 'cap-color': 4, 'bruises': 0, 'odor': 6, 'gill-spacing': 0, 'gill-size': 1, 'gill-color': 4, 'stalk-shape': 0, 'stalk-root': 3, 'stalk-surface-above-ring': 2, 'stalk-surface-below-ring': 2, 'stalk-color-above-ring': 7, 'stalk-color-below-ring': 7, 'ring-number': 1, 'ring-type': 4, 'spore-print-color': 2, 'population': 3, 'habitat': 5},
    {'cap-shape': 5, 'cap-surface': 2, 'cap-color': 9, 'bruises': 0, 'odor': 0, 'gill-spacing': 0, 'gill-size': 0, 'gill-color': 4, 'stalk-shape': 1, 'stalk-root': 2, 'stalk-surface-above-ring': 2, 'stalk-surface-below-ring': 2, 'stalk-color-above-ring': 7, 'stalk-color-below-ring': 7, 'ring-number': 1, 'ring-type': 4, 'spore-print-color': 3, 'population': 2, 'habitat': 1},
    {'cap-shape': 0, 'cap-surface': 2, 'cap-color': 8, 'bruises': 0, 'odor': 3, 'gill-spacing': 0, 'gill-size': 0, 'gill-color': 5, 'stalk-shape': 1, 'stalk-root': 2, 'stalk-surface-above-ring': 2, 'stalk-surface-below-ring': 2, 'stalk-color-above-ring': 7, 'stalk-color-below-ring': 7, 'ring-number': 1, 'ring-type': 4, 'spore-print-color': 3, 'population': 2, 'habitat': 3},
    {'cap-shape': 5, 'cap-surface': 3, 'cap-color': 8, 'bruises': 0, 'odor': 5, 'gill-spacing': 0, 'gill-size': 1, 'gill-color': 5, 'stalk-shape': 0, 'stalk-root': 3, 'stalk-surface-above-ring': 2, 'stalk-surface-below-ring': 2, 'stalk-color-above-ring': 7, 'stalk-color-below-ring': 7, 'ring-number': 1, 'ring-type': 4, 'spore-print-color': 2, 'population': 3, 'habitat': 5},
    {'cap-shape': 5, 'cap-surface': 2, 'cap-color': 3, 'bruises': 1, 'odor': 0, 'gill-spacing': 1, 'gill-size': 0, 'gill-color': 5, 'stalk-shape': 1, 'stalk-root': 1, 'stalk-surface-above-ring': 2, 'stalk-surface-below-ring': 2, 'stalk-color-above-ring': 7, 'stalk-color-below-ring': 7, 'ring-number': 1, 'ring-type': 0, 'spore-print-color': 3, 'population': 0, 'habitat': 1},
    {'cap-shape': 5, 'cap-surface': 3, 'cap-color': 9, 'bruises': 0, 'odor': 0, 'gill-spacing': 0, 'gill-size': 0, 'gill-color': 4, 'stalk-shape': 1, 'stalk-root': 2, 'stalk-surface-above-ring': 2, 'stalk-surface-below-ring': 2, 'stalk-color-above-ring': 7, 'stalk-color-below-ring': 7, 'ring-number': 1, 'ring-type': 4, 'spore-print-color': 2, 'population': 2, 'habitat': 1},
    {'cap-shape': 0, 'cap-surface': 2, 'cap-color': 8, 'bruises': 0, 'odor': 3, 'gill-spacing': 0, 'gill-size': 0, 'gill-color': 5, 'stalk-shape': 1, 'stalk-root': 2, 'stalk-surface-above-ring': 2, 'stalk-surface-below-ring': 2, 'stalk-color-above-ring': 7, 'stalk-color-below-ring': 7, 'ring-number': 1, 'ring-type': 4, 'spore-print-color': 2, 'population': 2, 'habitat': 3},
    {'cap-shape': 0, 'cap-surface': 3, 'cap-color': 8, 'bruises': 0, 'odor': 3, 'gill-spacing': 0, 'gill-size': 0, 'gill-color': 5, 'stalk-shape': 1, 'stalk-root': 2, 'stalk-surface-above-ring': 2, 'stalk-surface-below-ring': 2, 'stalk-color-above-ring': 7, 'stalk-color-below-ring': 7, 'ring-number': 1, 'ring-type': 4, 'spore-print-color': 3, 'population': 3, 'habitat': 3},
    {'cap-shape': 5, 'cap-surface': 3, 'cap-color': 9, 'bruises': 0, 'odor': 6, 'gill-spacing': 0, 'gill-size': 1, 'gill-color': 4, 'stalk-shape': 0, 'stalk-root': 3, 'stalk-surface-above-ring': 2, 'stalk-surface-below-ring': 2, 'stalk-color-above-ring': 7, 'stalk-color-below-ring': 7, 'ring-number': 1, 'ring-type': 4, 'spore-print-color': 2, 'population': 4, 'habitat': 1},
    {'cap-shape': 0, 'cap-surface': 2, 'cap-color': 3, 'bruises': 1, 'odor': 0, 'gill-spacing': 1, 'gill-size': 0, 'gill-color': 5, 'stalk-shape': 1, 'stalk-root': 1, 'stalk-surface-above-ring': 2, 'stalk-surface-below-ring': 2, 'stalk-color-above-ring': 7, 'stalk-color-below-ring': 7, 'ring-number': 1, 'ring-type': 0, 'spore-print-color': 2, 'population': 3, 'habitat': 3}
])
newpred = model.predict(new_data)
for i, pred in enumerate(newpred, 1):
    print(f"{i}번째 예측 결과 :", "독버섯입니다." if pred == 1 else "식용 버섯입니다.")
print("확률은 \n", model.predict_proba(new_data))