'''
앙상블(Ensemble) 학습
    여러개의 분류기를 생성하고, 그 예측을 결합함으로써 보다 정확한 예측결과를 얻음
    강력한 하나의 모델보다는 약한 모델 여러개를 조합하여 더 정확한 예측결과를 도출

위스콘신 유방암 데이터 세트(Wisconsin breast cancer dataset) 사용
    분류데이터로 많이 사용하는 데이터셋중 하나.
    diagnosis = 종속변수
    The diagnosis of breast tissues (M = malignant, B = benign)

'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  #(왜곡데이터니까)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline # 전처리랑 모델을 일체형 객체로 관리
from sklearn.ensemble import VotingClassifier
# 앙상블에 사용할 모델 불러오기
from sklearn.linear_model import LogisticRegression # 로지스틱회귀
from sklearn.tree import DecisionTreeClassifier     # 의사결정 트리
from sklearn.neighbors import KNeighborsClassifier  # KNN
# 정확도 확인
from sklearn.metrics import accuracy_score
# 데이터 갯수 확인
from collections import Counter

# 데이터 가져오기
cancer = load_breast_cancer()
print(cancer.keys())
x, y = cancer.data, cancer.target
print(x[:2])
print(y[:2])
print('diagnosis(y) :', np.unique(y)) #  [0:악성(M) 1:양성(B)]

# 0과 1의 비율 - StratifiedKFold를 사용하려고
counter = Counter(y)
total = sum(counter.values())
for cls, cnt in counter.items():
    print(f'class : {cls} - {cnt}개 ({cnt / total:.2%})')
# class : 0 - 212개 (37.26%)
# class : 1 - 357개 (62.74%)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12, 
                                                    stratify=y) 
#stratify=y => train, test 비율유지 : 불균형데이터 모델평가시 왜곡방지
y_li = y.tolist()
ytr_li  = y_train.tolist()
yte_li  = y_test.tolist()

print("전체 분포 :", Counter(y_li))     # Counter({1: 357, 0: 212})
print("train 분포 :", Counter(ytr_li))  # Counter({1: 285, 0: 170})
print("test 분포 :", Counter(yte_li))   # Counter({1: 72, 0: 42})
print()

# 개별 모델 생성하기 - 로지스틱, 의사결정나무, kNN
# make_pipeline((StandardScaler, LogisticRegression()) - 표준화후 로지스틱모델생성
logi = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='lbfgs', random_state=12)) 
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
tree = DecisionTreeClassifier(max_depth=5, random_state=12)

# 앙상블 모델 만들기 - VotingClassifier
voting = VotingClassifier(
    estimators=[('LR', logi), ('KNN', knn), ('DT', tree)], 
    voting='soft'
)

# 개별 모델 성능 확인
named_models = [('LR', logi), ('KNN', knn), ('DT', tree)]
for name, clf in named_models:
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(f'{name} 정확도 : {accuracy_score(y_test, pred):.4f}')

# LR 정확도 : 0.9912
# KNN 정확도 : 0.9737
# DT 정확도 : 0.8772
print()

# Voting(앙상블)의 성능 평가
voting.fit(x_train, y_train)
vpred = voting.predict(x_test)
print(f"Voting 분류기 정확도 : {accuracy_score(y_test, vpred):4f}") # 0.964912

# 선택 : 레이블이 한쪽으로 왜곡되어 있으니까 선택적으로 교차검증으로 안정성을 확인할 필요가 있다
cvfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=12) 
# shuffle=True => 왜곡되어 있기 때문에 섞어야한다.
cv_score = cross_val_score(voting, x, y, cv=cvfold, scoring='accuracy')
print(f'Voting 5겹짜리 CV 정확도 평균 : {cv_score.mean():.4f}')       # 0.9701 -> 조금 나아진다
print(f'Voting 5겹짜리 CV 정확도 표준편차 : +- {cv_score.std():.4f}') # +- 0.0181
print()

# 모델 성능 지표
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print('Voting Model 상세 평가 -------------------------------')
print(classification_report(y_test, vpred))
print('confusion_matrix : \n', confusion_matrix(y_test, vpred))
#  [[40    2]
#  [ 2    70]]
print('roc_auc_score :',roc_auc_score(y_test, voting.predict_proba(x_test)[:, 1])) # 0.994