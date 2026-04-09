'''
[과적합(Overfitting) 방지] - 분류모델에서 엄청 신경 써야한다!
    1) train-test split
        from sklearn.model_selection import train_test_split
        일반화 성능 향상
        과적합 여부 확인

    2)K-Fold Cross-Validation (k-겹 교차 검증)
        from sklearn.model_selection import KFold, cross_val_score(단순화K-Fold)
        train-test split을 써도 과적합이 나오면 사용
        머신러닝 모델의 성능 평가 및 검증을 위한 방법
        train data를 분할해 학습과 평과를 병행하는 방법 
        대표적인 방법으로 K-Fold가 가장 일반적 
        안정적 평가가 목적
        참고 : from sklearn.model_selection import StratifiedKFold
            StratifiedKFold
                불균형한 분포도를 가진 label데이터 집합(편향된, 왜곡된)을 
                처리하기 위한 k-fold방식
                ex) 대출 사기 데이터인 경우 대부분은 정상, 사기 레이블은 극히 일부임

    3)GridSearchCV
        from sklearn.model_selection import GridSearchCV
        과적합 방지 '간접' 방법
        최적의 파라미터 찾기(최적의 Hyper Parameter 검색)
            내부적으로 KFold사용해 과적합을 줄이는데 도움을 줌
        연습용으로 일부 파라미터만 사용 :
            max_depth, 
            min_samples_split - 노드 분할을 위한 최소한의 샘플수로 과적합 제어
        
    4) L1, L2 규제 - 정규화
    
    5) 그외
        - 불필요한 변수 제거
        - 데이터량 증가
        - 조기종료

    iris dataset 사용
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# iris dataset 가져와서 추출
iris = load_iris()
print(iris.keys())
train_data = iris.data      # x
train_lable = iris.target   # y
print(train_data[:3])
print(train_lable[:3])
print()

# 분류 모델 작성
dt_clf = DecisionTreeClassifier()
dt_clf.fit(train_data, train_lable) # 모든 데이터 학습에 참여 시킴
pred = dt_clf.predict(train_data)   # 학습 데이터로 검증(예측)
print('예측값 : ',pred)
print('실제값 : ',train_lable)
print('분류 정확도 :', accuracy_score(train_lable, pred)) # 1.0
print()
# => 학습데이터로 검증을 했기때문에 과적합 의심

#==============================================================================
# 과적합(Overfitting) 방지 처리 1) train-test split
#   과적합 여부 확인
#==============================================================================
print('-'*20,'과적합(Overfitting) 방지 처리 1) train-test split','-'*20)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, 
                                                    test_size=0.3,
                                                    random_state=121)
dt_clf.fit(x_train, y_train)    # train data로 학습
pred2 = dt_clf.predict(x_test)  # test data로 검증(예측)
print('예측값 : ',pred2)
print('실제값 : ',y_test)
print('분류 정확도 :', accuracy_score(y_test, pred2)) # 0.95555
# 효과 => 과적합 여부가 확인이 됨
print()


#==============================================================================
# 과적합(Overfitting) 방지 처리 2-1) 교차검증(Cross Validation)
# train data를 분할(k개 만큼)해 학습과 평과를 병행하는 방법 
# 대표적인 방법으로 K-Fold가 가장 일반적 
# 참고 : from sklearn.model_selection import StratifiedKFold
#       StratifiedKFold : 불균형한 분포도를 가진 label데이터 집합(편향된, 왜곡된)을 처리하기 위한
#                       k-fold방식  - ex) 대출 사기 데이터인 경우 대부분은 정상, 사기 레이블은 극히 일부임
#==============================================================================
print('-'*20,'과적합(Overfitting) 방지 처리 2-1) 교차검증(Cross Validation)','-'*20)
from sklearn.model_selection import KFold

# 편의상 전체 데이터 사용
features = iris.data
label = iris.target
dt_clf2 = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=12)


kflod = KFold(n_splits=5) # k: 5회 접기

# 과정 기록용 리스트
cv_acc = []
print('iris shape :', features.shape) # (150, 4)
print()

# k-fold 학습시 전체 150행이 학습데이터(4/5, 120개), 검증데이터(1/5, 30개)로 분할되어 학습함
# k -fold 객체의 split()을 호출하면 Fold별 학습용, 검증용 테스트의 행인덱스를 arrary로 변환
n_iter = 0
for train_index, test_index in kflod.split(features):
    # # 쪼개짐 확인
    # print('n_iter(반복 수 - 쪼개짐확인) : ', n_iter)
    # print('train_index : ', train_index)
    # print('test_index : ', test_index) # test data를 먼저 쪼개고 나머지는 train
    # n_iter += 1

    xtrain, xtest, = features[train_index], features[test_index]
    ytrain, ytest = label[train_index], label[test_index]

    # 학습 및 예측
    dt_clf2.fit(xtrain, ytrain)     # train으로 학습
    pred = dt_clf2.predict(xtest)   # test로 검증
    n_iter += 1
    
    # 반복할 때 마다 정확도 출력하기
    acc = np.round(accuracy_score(ytest, pred),5)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print(f"반복수: {n_iter}, 교차검증 정확도 : {acc},\
    학습데이터 크기 : {train_size}, 검증데이터 크기 : {test_size}")
    print(f'반복수:{n_iter}, 검증데이터 인덱스{test_index}')
    cv_acc.append(acc)
    print()

print()
print("cv_acc :", np.array(cv_acc).astype(float)) # [1.  0.96667 0.83333 0.93333 0.73333]
print('평균 검증 정확도 :', np.mean(cv_acc))
print()

#==============================================================================
# 과적합(Overfitting) 방지 처리 2-2) 교차검증(Cross Validation) 단순화
# cross_val_score를 이용해 교차검증을 간단히 처리가능
#   내부적으로 KFold으로 처리함
#==============================================================================
print('-'*20,'과적합(Overfitting) 방지 처리 2-2) 교차검증(Cross Validation) 단순화','-'*20)
from sklearn.model_selection import cross_val_score

data = iris.data
label = iris.target

score = cross_val_score(dt_clf2, data, label, scoring='accuracy', cv=5)
print("교차 검증별 정확도 :", np.round(score, 3))
print('평균 검증 정확도 :',np.round(np.mean(score),3) )
print()

#==============================================================================
# 과적합(Overfitting) 방지 처리 3) GridSearchCV
#   과적합 방지 '간접' 방법
#   최적의 파라미터 찾기 : 내부적으로 KFold사용해 과적합을 줄이는데 도움을 줌
#   연습용으로 일부 파라미터만 사용 :
#       max_depth : 노드의 최대 깊이 
#       min_samples_split : 노드 분할을 위한 최소한의 샘플수로 과적합 제어
#==============================================================================
print('-'*20,'과적합(Overfitting) 방지 처리 3) GridSearchCV','-'*20)
from sklearn.model_selection import GridSearchCV

# 최적의 모델을 찾기위해  값을 줌 - GridSearchCV를 6(3*2)번을 알아서 돌려줌
parameters = {'max_depth' : [1, 2, 3], 'min_samples_split' : [2, 3]} 
grid_dtree = GridSearchCV(estimator=dt_clf2, # estimator :실험용 모델
                            param_grid=parameters,
                            cv=3,       # KFold
                            refit=True) # 재학습

# 내부적으로 복수 개의 모형을 생성하고 이를 시행시켜 최적의 parameter를 찾아줌.
grid_dtree.fit(x_train, y_train) 

# GridSearchCV의 cv_results_값들 :
#   best_scroe_, best_params_. best_esmator_, grid_score_...이 있다.
import pandas as pd
pd.set_option('display.max_columns', None)
scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df)

print("GridSearchCV의 최적의 파라미터 :", grid_dtree.best_params_)  # {'max_depth': 3, 'min_samples_split': 2}
print("GridSearchCV의 최적의 정확도 :", grid_dtree.best_score_)     # 0.9428571428571427
print()

# 최적의 모델 - 최적의 Parameter로 모델 생성
bestmodel = grid_dtree.best_estimator_
print(bestmodel) # DecisionTreeClassifier(max_depth=3, random_state=12)
best_pred = bestmodel.predict(x_test)
print("예측 결과 : ",best_pred)
print("분류 정확도 : ",accuracy_score(y_test, best_pred)) # 0.955
