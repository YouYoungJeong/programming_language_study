'''
[Randomforest 문제1] 
kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)
dataset은 winequality-red.csv 
https://www.kaggle.com/sh6147782/winequalityred?select=winequality-red.csv
Input variables (based on physicochemical tests):
        1 - fixed acidity
        2 - volatile acidity
        3 - citric acid
        4 - residual sugar
        5 - chlorides
        6 - free sulfur dioxide
        7 - total sulfur dioxide
        8 - density
        9 - pH
        10 - sulphates
        11 - alcohol
        Output variable (based on sensory data):
        12 - quality (score between 0 and 10)
'''
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from scipy import stats

from sklearn.datasets import fetch_openml # sklearn이 만든 연습용 dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline # 전처리 + 모델을 하나로 묶어서 실행.
from sklearn.compose import ColumnTransformer # Column별 전처리를 다르게 처리할때 사용하는 클래스
from sklearn.impute import SimpleImputer # 결측치 처리할 때 사용하면 좋다.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns


# 데이터 확인하기
wine = pd.read_csv("winequality-red.csv")
print(wine.head(3))
print(wine.info())
print(set(wine['quality'])) # {3, 4, 5, 6, 7, 8}
print(wine.isnull().sum())  # 0

# 데이터 추출하기
x = wine.drop('quality', axis=1)
y = wine['quality']
print(x.head(2))
print(y.head(2))

# x 데이터 범위 확인하기(이상치)
plt.figure(figsize=(20, 10))
sns.boxenplot(x)
plt.show()
plt.close()
'''
total sulfur dioxide 이상치
'''

# 이상치 처리하기
print(x['total sulfur dioxide'].describe())

# 이상치 boxplot으로 확인하기
sns.boxenplot(x['total sulfur dioxide'])
plt.show()
plt.close()

q1 = x['total sulfur dioxide'].quantile(0.25)
q2 = x['total sulfur dioxide'].quantile(0.50)
q3 = x['total sulfur dioxide'].quantile(0.75)

iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outlier_df = x[(x['total sulfur dioxide'] < lower) | (x['total sulfur dioxide'] > upper)]
normal_df = x[(x['total sulfur dioxide'] >= lower) & (x['total sulfur dioxide'] <= upper)]
print(outlier_df['total sulfur dioxide'])
outlier_count = ((x['total sulfur dioxide'] < lower) | (x['total sulfur dioxide'] > upper)).sum()
print("이상치 개수 :", outlier_count) # 54

# 이상치 값 평균값으로 대체
mean_value = x['total sulfur dioxide'].mean()
x['total sulfur dioxide'] = x['total sulfur dioxide'].apply(
    lambda x: mean_value if x < lower or x > upper else x)

# 바뀐 값 boxplot으로 확인하기
sns.boxenplot(x['total sulfur dioxide'])
plt.show()
plt.close()

# train-test scaleing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (1117, 11) (479, 11) (1117,) (479,)

# 모델 생성
model = RandomForestClassifier()

# 하이퍼파라미터 튜닝 범위 설정하기
param_grid = {
    'n_estimators':[1000, 2000],        # d_tree 갯수
    'max_depth' : [5, 10, None],      # tree의 깊이
    'class_weight':[None,'balanced']  # 클래스 불균형 보정의 유무
}

# 불균형한 데이터 K-Fold 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 12)

# GridSearchCV 진행
grid = GridSearchCV(
    model,           
    param_grid,         # 탐색할 파라미터
    cv=cv,              # 교차검증
    scoring='accuracy',  # 평가기준
    n_jobs=-1           # cpu사용 갯수 : -1(모든 CPU다 사용하겠다.)
)

# 최적의 파라미터 탐색 + 학습수행
grid.fit(x_train, y_train)
print("최적의 파라미터 :",  grid.best_params_)
# {'class_weight': None, 'max_depth': None, 'n_estimators': 2000}
print("최적의 모델 :",  grid.best_estimator_) # RandomForestClassifier(n_estimators=2000)
print("최적의 점수 :", grid.best_score_) # 0.6812259769378605

# 특정 변수를 대상으로 중요 변수 확인하기
n_features = x.shape[1]
importances = grid.best_estimator_.feature_importances_
indices = np.argsort(importances)
plt.barh(range(n_features), grid.best_estimator_.feature_importances_, align='center')
plt.title("특정 변수를 대상으로 중요 변수 확인하기")
plt.xlabel('Fearture 중요도(importances score)')
plt.ylabel('Feartures')
plt.yticks(range(n_features), x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()

# 새로운 와인 데이터가 들어왔을 때 품질 예측
new_wine = [[7.5, 0.5, 0.3, 2.0, 0.08, 15.0, 50.0, 0.997, 3.3, 0.6, 10.5]]
new_pred = grid.predict(new_wine)
print(f"새로운 와인 샘플의 예측 품질 등급: {new_pred[0]}")
# ============================================================================
