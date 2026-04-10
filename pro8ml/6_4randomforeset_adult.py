'''
앙상블(Ensemble) 모델 랜덤 포레스트 (Random Forest) 분류 알고리즘
sklearn 각 모듈 정리! 이자료는 실무화를 진행함.
    adult dataset 사용하기
        성인 소득 예측자료
        연봉이 50K(약 $50,000) 이상인지 예측 - 이진분류

            #   Column          Non-Null Count  Dtype   
        ---  ------          --------------  -----   
        0   age             48842 non-null  int64   
        1   workclass       46043 non-null  category
        2   fnlwgt          48842 non-null  int64   
        3   education       48842 non-null  category
        4   education-num   48842 non-null  int64   
        5   marital-status  48842 non-null  category
        6   occupation      46033 non-null  category
        7   relationship    48842 non-null  category
        8   race            48842 non-null  category
        9   sex             48842 non-null  category
        10  capital-gain    48842 non-null  int64   
        11  capital-loss    48842 non-null  int64   
        12  hours-per-week  48842 non-null  int64   
        13  native-country  47985 non-null  category
        14  class           48842 non-null  category
        dtypes: category(9), int64(6)
'''
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.datasets import fetch_openml # sklearn이 만든 연습용 dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline # 전처리 + 모델을 하나로 묶어서 실행.
from sklearn.compose import ColumnTransformer # Column별 전처리를 다르게 처리할때 사용하는 클래스
from sklearn.impute import SimpleImputer # 결측치 처리할 때 사용하면 좋다.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# data
adult = fetch_openml(name='adult', version=2, as_frame=True)
print(type(adult)) # <class 'sklearn.utils._bunch.Bunch'> 형태 변환 해줘야함

df = adult.frame
print(df.head())
print(df.shape) # (48842, 15)
print(df.info()) # native-country-nan, category(9), int64(6)

# target 변환(인코딩): class(연봉) <=50K : 0 , >50K : 1
df['class'] = df['class'].apply(lambda class_value : 1 if '>50K' in class_value else 0)
print(df.head(3))
print(set(df['class'])) # {0, 1}

x = df.drop('class', axis=1)    # feature
y = df['class']                 # label
print(x.info())

# 컬럼 분리 : 숫자형, 범주형
num_cols = x.select_dtypes(include=['int64','float64']).columns # 숫자형 칼럼만 선택
cat_cols = x.select_dtypes(include=['category', 'object']).columns # 범주형 칼럼만 선택

# ML에서의 Pipeline 사용법
# 전처리 파이프라인(숫자형에 대해서) - 처리항목들을 연결해 연속적을 실행
num_pipeline = Pipeline([
    
    # 결측치처리 - NaN값 중앙값으로 채우기
    ('imputer', SimpleImputer(strategy='median')),
    # 표준화 - 평균0, 표준편차1인 분산화
    ('scaler', StandardScaler())
])

# 전처리 파이프라인(범주형에 대해서) - 처리항목들을 연결해 연속적을 실행
cat_pipeline = Pipeline([
    
    # 결측치처리 - NaN값 최빈값으로 채우기(범주형이니까 숫자계산이 불가)
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # 범주형의 완화 처리
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 컬럼별 전처리 결합
# 숫자는 Scaling, 문자는 Encoding : 각각 다르게 처리해야함.
# ColumnTransformer 칼럼별 처리하는 방법이 다를때 사용
preprocess = ColumnTransformer([
    
    # 숫자형 칼럼에 num_pipeline 적용
    ('num' , num_pipeline, num_cols),
    # 범주형 칼럼에 num_pipeline 적용
    ('cat' , cat_pipeline, cat_cols)
])

# 전체 파이프 (전처리 + 모델) - 처리를 위한 마지막 준비
pipeline = Pipeline([
    # 전처리 하기
    ('prep', preprocess),
    # 모델 만들기
    ('model', RandomForestClassifier(random_state=12))
])

# train / test split
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.3, random_state=12,stratify=y # 클래스 비율을 유지하겠다
    )

# 하이퍼파라미터 튜닝 범위 설정하기
param_grid = {
    'model__n_estimators':[100, 200],        # d_tree 갯수
    'model__max_depth' : [5, 10, None],      # tree의 깊이
    'model__class_weight':[None,'balanced']  # 클래스 불균형 보정의 유무
}

# 불균형한 데이터 K-Fold 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 12)

# GridSearchCV 진행
grid = GridSearchCV(
    pipeline,           # 전체 파이프라인 사용
    param_grid,         # 탐색할 파라미터
    cv=cv,              # 교차검증
    scoring='roc_auc',  # 평가기준
    n_jobs=-1           # cpu사용 갯수 : -1(모든 CPU다 사용하겠다.)
)

# 전처리 + 최적의 파라미터 탐색 + 학습수행
grid.fit(train_x, train_y)
print("최적의 파라미터 :",  grid.best_params_)
# {'model__class_weight': None, 'model__max_depth': 10, 'model__n_estimators': 200}
print("최적의 모델 :",  grid.best_estimator_)
'''
Pipeline
    ├─ prep : ColumnTransformer
    │   ├─ num : 결측치 중앙값 대체 → 표준화
    │   └─ cat : 결측치 최빈값 대체 → 원핫인코딩
    └─ model : RandomForestClassifier(max_depth=10, n_estimators=200, random_state=12)

최적의 모델 : 
Pipeline(
    steps=[
        (
            'prep',
            ColumnTransformer(
                transformers=[
                    (
                        'num',
                        Pipeline(
                            steps=[
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
                            ]
                        ),
                        Index(
                            ['age', 'fnlwgt', 'education-num',
                            'capital-gain', 'capital-loss', 'hours-per-week'],
                            dtype='object'
                        )
                    ),
                    (
                        'cat',
                        Pipeline(
                            steps=[
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))
                            ]
                        ),
                        Index(
                            ['workclass', 'education', 'marital-status',
                            'occupation', 'relationship', 'race',
                            'sex', 'native-country'],
                            dtype='object'
                        )
                    )
                ]
            )
        ),
        (
            'model',
            RandomForestClassifier(
                max_depth=10,
                n_estimators=200,
                random_state=12
            )
        )
    ]
)
'''

# 예측하기
pred = grid.predict(test_x)
proba =grid.predict_proba(test_x)[:, 1] # class : 1 에 대한 확률값

# 평가하기
print("전체 정확도 :", accuracy_score(test_y ,pred))    # 0.857367
print("ROC_AUC Score: ", roc_auc_score(test_y, proba)) # 0.91059
print("classification_report\n",classification_report(test_y, pred))
'''
classification_report
                precision    recall  f1-score   support

        0       0.87      0.95      0.91     11147
        1       0.79      0.55      0.65      3506

    accuracy                           0.86     14653
    macro avg      0.83      0.75      0.78     14653
weighted avg       0.85      0.86      0.85     14653
'''