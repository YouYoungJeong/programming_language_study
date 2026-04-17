''' ex50
sklearn 제공 Regressior성능 비교  (Regression(회귀) 정리)

Regression model 판단
    pipline + GridSearchCV + 교차검증 + 성능확인 + 시각화

diabetes 데이터 설명
Pregnancies: 임신 여부
Glucose: 포도당 수치
BloodPressure: 혈압
SkinThickness: 피부의 두께
Insulin: 인슐린 수치
BMI: 체질량 지수, 체중을 신장의 제곱으로 나눈 값
DiabetesPedigreeFunction: 당뇨 혈통 함수
Age: 나이
Diabetes: 당뇨 여부, 예측 목표값
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.datasets import load_diabetes
# 전처리, GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline # 실무에서 진짜 많이 사용함!
from sklearn.preprocessing import StandardScaler
# 모델
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
# 성능평가
from sklearn.metrics import r2_score, mean_squared_error

data = load_diabetes()
x = data.data
y = data.target
print(x[:2])    # [[ 0.03807591  0.05068012  ,,,
print(y[:2])    # [151.  75.] 연속형

# 데이터 분할 train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Pipline(앙상블 기법) + GridSearchCV 변수 생성하기
models = {
    "LinearRegression" : {
        "pipeline" : Pipeline([
            ('scaler', StandardScaler()),
            ("model", LinearRegression())
        ]),
    "params":{ # GridSearchCV를 쓰기위한 param정의
        "model__fit_intercept":[True, False] # LinearRegression 옵션
        }
    },
    "RandomForestRegressor" : {
        "pipeline" : Pipeline([
            ("model", RandomForestRegressor(random_state=42))
        ]),
    "params":{ # RandomForestRegressor옵션
        "model__n_estimators":[100, 200],
        "model__max_depth":[None, 5, 10],
        "model__min_samples_split":[2,5]
        }
    },
    "XGBRegressor" : {
        "pipeline" : Pipeline([
            ("model", XGBRegressor(random_state=42, verbosity=0))
        ]),
    "params":{ # XGBRegressor 옵션
        "model__n_estimators":[100, 200],
        "model__learning_rate":[0.01, 0.05], # 학습률
        "model__max_depth":[3,5]
        }
    },
    "SVR" : {
        "pipeline" : Pipeline([
            ('scaler', StandardScaler()),
            ("model", SVR())
        ]),
    "params":{ # SVR 옵션
        "model__C":[0.1, 1, 10],
        "model__gamma":["scale","auto"], # defualt=auto
        "model__kernel":["rbf"]
        }
    },
    "KNeighborsRegressor" : {
        "pipeline" : Pipeline([
            ('scaler', StandardScaler()), # knn은 스케일 절대적으로 필요
            ("model", KNeighborsRegressor())
        ]),
    "params":{ # KNeighborsRegressor 옵션
        "model__n_neighbors":[3, 5, 7],
        "model__weights":["uniform","distance"]
        }
    }
}

# GridSearchCV 실행
results = []
best_models = {}

# 각 model을 순서대로 반복 처리 : best모델 추출(생성), 성능을 저장
for name, config in models.items():
    print(f'{name} 튜닝중....')
    grid = GridSearchCV(
        config['pipeline'],
        config['params'],
        cv = 5,
        scoring="r2",
        n_jobs= -1
    )
    # 학습하기
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_ # GridSearchCV가 제공하는 최고의 모델
    best_pred = best_model.predict(x_test)

    # 성능 평가
    rmse = np.sqrt(mean_squared_error(y_test, best_pred)) # RMSE
    r2 = r2_score(y_test, best_pred)
    
    # 담기
    results.append([name, rmse, r2])
    best_models[name] = best_model
    # 출력하기
    print(f"best_params : {grid.best_params_}")
    print(f"R2(결정계수-설명력) : {r2}")

# 최종결과를 DataFrame에 저장.
df_resutls = pd.DataFrame(results, columns=["model_name",'rmse','r2'])
df_resutls = df_resutls.sort_values("r2", ascending=False)
print("\n최종 성능 비교")
print(df_resutls)

# 모델 성능 비교를 위한 시각화 하기
plt.figure(figsize=(12, 5))

# r2
plt.subplot(1, 2, 1)
sns.barplot(x="model_name",y='r2', data=df_resutls)
plt.xlabel("Model")
plt.ylabel("R²")
plt.title("튜닝 모델의 R²(결정계수)")
plt.xticks(rotation=30)

# RMSE
plt.subplot(1, 2, 2)
sns.barplot(x="model_name",y='rmse', data=df_resutls)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("튜닝 모델의 RMSE")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# Best Model 예측 시각화
# 1) 최고 모델을 선택 - SVR이 1등
best_model_name = df_resutls.iloc[0]['model_name']
best_model = best_models[best_model_name]
best_model_pred = best_model.predict(x_test)

#2) 시각화
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_model_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'최고 모델 : {best_model_name}')
plt.xlabel("실제값")
plt.ylabel("예측값")
plt.show()