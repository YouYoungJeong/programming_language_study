'''
앙상블(Ensemble) 모델 랜덤 포레스트 (Random Forest) 분류 알고리즘 Regression
    랜덤 포레스트 (Random Forest)는 분류(Classification), 회귀(Regression) 모두가능

캘리포니아 주택 가격 데이터셋 사용
dataset_housing = datasets.fetch_california_housing()
주요 독립 변수:
    MedInc: 중간 소득.
    HouseAge: 주택의 중간 연령.
    AveRooms: 가구당 평균 방 개수.
    AveBedrms: 가구당 평균 침실 개수.
    Population: 지역 내 총 인구 수.
    AveOccup: 가구당 평균 거주 인원.
    Latitude, Longitude: 지역의 위도와 경도.
종속 변수:
    target: 지역의 중간 주택 가격(단위: $100,000).
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error # 모델의 성능 파악

pd.set_option('display.max_columns', None)

housing = fetch_california_housing(as_frame=True) # as_frame=True DF 형태로 반환
print(housing.DESCR) # 데이터에 대한 설명
print(housing.data[:2])
print(housing.target[:2])
print(housing.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
df = housing.frame
print(df.head(3))
print(df.info())
print()

# x, y자료 추출
x = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

# 모델생성하기
rfmodel = RandomForestRegressor(n_estimators=200, random_state=42)
rfmodel.fit(x_train, y_train)

# 예측및 평가하기
y_pred = rfmodel.predict(x_test)
print(f'MSE : {mean_squared_error(y_test, y_pred):.3f}') # 0.254
print(f"R² : {r2_score(y_test, y_pred):.3f}")            #  0.807
print()

# 변수 기여도(중요도) 시각화
importances = rfmodel.feature_importances_ 
# Indexes == indices
indices = np.argsort(importances)[::-1] # argsort()-오름차순 ,argsort()[::-1] - 내림차순
plt.figure(figsize=(8, 5))
plt.bar(range(x.shape[1]), importances[indices])
plt.xticks(range(x.shape[1]), x.columns[indices], rotation=45)
plt.title("변수 기여도(중요도) 시각화")
plt.xlabel("feature name")
plt.ylabel("feature importances")
plt.tight_layout()
plt.show()

# 중요 변수 순위정보 저장
print('중요 변수 순위정보 저장')
ranking = pd.DataFrame({
    'feature' : x.columns[indices],
    "importances" : importances[indices]
})
print(ranking)
print()

# 파라미터 튜닝 - RandomizedSearchCV
# GridSearchCV와 달리 사용자가 지정한 범위, 분포에서 임의로 일부 혼합만 샘플링해 탐색
# 연속적 값 범위도 가능. 단점 : (전문적일 때)무작위 이기 때문에 최적 조합을 못찾을 수 있다.
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators':[200, 400, 800],
    'max_depth':[None, 10, 20, 30],
    'min_samples_leaf':[1, 2, 4], # 마지막 노드의 필요한 최소 샘플수
    'min_samples_split':[1, 2, 4], # 노드 분할에 필요한 최소 샘플수
    'max_features':[None, 'sqrt', 'log2',1.0, 0.8, 0.6] # 분할 시 고려할 최대 특성수
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,      # 20개의 random parameter 조합함.
    scoring='r2',
    cv=3,
    random_state=42,
    verbose=1       # 진행상황을 출력함(DL에서 사용함)
)
# 탐색 학습
search.fit(x_train, y_train)
print("search.best_params_ :", search.best_params_)
best = search.best_estimator_
print('best_score_ :', search.best_score_)
print('final R² :', r2_score(y_test, best.predict(x_test)))