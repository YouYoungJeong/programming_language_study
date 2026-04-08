'''
Logistic Linear Regression(로지스틱회귀분석) : 다항분류 - iris dataset
소프트맥스(Softmax) 함수
    다중 클래스 분류 모델의 출력층에서 사용되는 활성화 함수로, 
    여러 개의 실수값을 0~1 사이의 확률값으로 정규화하여 합이 1이 되도록 변환
from sklearn.linear_model import LogisticRegression 
    LogisticRegression : 다중(class, 종속변수)를 지원하도록 일반화됨
    softmax regression, multinormial logistic regression 이라고 부른다. 

Sigmoid: 2개의 클래스(이진 분류)를 대상으로 하는 경우.
Softmax: 2개 이상의 다중 클래스를 대상으로 하는, Sigmoid의 일반화된 형태.

데이터 크기 표준화(Scaling)
    최적화 과정에서 안정성, 수렴속도 향상, 과적합/과소적합 방지등의 효과가 있음
    데이터 크기의 차이가 큰 경우 사용한다
'''
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split # 모델 샘플링 추출 모듈
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # 표준화
from sklearn.linear_model import LogisticRegression # softmax reg, multinormial logistic reg
import statsmodels.api as sm
import statsmodels.formula.api as smf

# sklearn datasets 읽는 방법
iris = datasets.load_iris()
print(iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
print(iris.target_names)
print(iris.target)
print(iris.feature_names)
print(iris.data[:3])
print(np.corrcoef(iris.data[:,2], iris.data[:,3])[0,1]) # 0.96286
print()

# 변수 추출
print('iris data 추출----------------------------------------------------')
x = iris.data[:, [2,3]] 
print(x.shape)# (150, 2)
y = iris.target
print(y.shape) # (150,)
print(x[:3])
print(y[:3])
print(set(map(int, y))) # set 숫자만 추출해서 보는 방법
print()

# train - test 
print("train_test_spilt (7 : 3)-------------------------------------------")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print('x_train :',x_train.shape,'\n', x_train[:3]) # (105, 2)
print('y_train :',y_train.shape,y_train[:3]) # (105,)
print('x_test :',x_test.shape,'\n', x_test[:3]) # (45, 2)
print('y_test :',y_test.shape, y_test[:3]) # (45,)
print()

# Scaling - 독립변수(feature)만 표준화 진행, 종립변수는 범주형(0,1)인데 표준화 왜해~
print("Scaling------------------------------------------------------------")
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3] ,'\n', x_test[:3])
print()

# Scaling 결과 원복
print('Scaling 결과 원복--------------------------------------------------')
x_train = sc.inverse_transform(x_train)
x_test = sc.inverse_transform(x_test)
print(x_train[:3],'\n',x_test[:3])
# 하지만 iris dataset의 크기의 차이가 없으므로 표준화는 의미 없다.
print()

# 분류모델 생성
print('분류 모델 생성-----------------------------------------------------')
# model = LogisticRegression(random_state=0)
model = LogisticRegression(random_state=0, C=100.0, solver='lbfgs',multi_class='multinomial')
# C : L2규제-모델에 패널티 적용1, 숫자값을 조정해 가며 정확도 확인
# solver : L1규제 -최적화 알고리름, 모델에 패널티 적용,solver='lbfgs'최적화를 위한 알고리즘 지원 
# multi_class : multinomial(다항분류)- softmax지원(auto로 하면 알아서 바뀜)
print(model)
print()

# 학습시키기
model.fit(x_train, y_train)

# 분류 예측
y_pred = model.predict(x_test)
print("예측값 :", y_pred)
print("실제값 :", y_test)
print()

print(f'총 갯수 : {len(y_test)}, 오류 수 : {np.sum(y_test != y_pred)}')
# test data 총 갯수 : 45, 오류 수 : 1
print()

# 분류 정확도 확인 1
print("분류 정확도 확인 1 sklearn accuracy_score 사용 ------------------")
print(f'{accuracy_score(y_test, y_pred)}') # 0.977777
print()

# 분류 정확도 확인 2
print("분류 정확도 확인 2 pandas crosstable사용 ------------------------")
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측치'], colnames=['관측치'])
print(con_mat)
print((con_mat[0][0]+con_mat[1][1]+con_mat[2][2]) / len(y_test)) #0.97777
print()

# 분류 정확도 확인 3
print("분류 정확도 확인 3 model.score함수 사용--------------------------")
print('test score : ', model.score(x_test, y_test))    # 0.97777
print('train score : ', model.score(x_train, y_train)) # 0.97142
# test score와 train score의 차이가 크면 overfitting 의심
print()

# 학습 후 검증이 된 모델 저장 후 읽기
import joblib  # pickle보다 빠르고 대용량을 지원

# 저장 - 이거 까지가 분석가가한다. 이후부터는 모델만 뿌려
joblib.dump(model, 'logimodel.pkl') # 확장자명은 sav, model... 이 3개가 일반적으로 많이 사용
del model

# 읽기
read_model = joblib.load('logimodel.pkl')

# 새로운 값으로 예측하기
# 주의! 만약 표준화된 자료로 모델을 생성했다면 new_data도 표준화 해야한다.
# sc.fit(new_data)
# new_data = sc.transform(new_data)
print("새로운 값으로 예측하기------------------------------------------")
new_data = np.array([[5.5, 2.2],[0.6, 0.3],[1.1, 0.5]])
new_pred = read_model.predict(new_data) 
# softmax가 준 확률값(각 범주 확률값의 합은 1) 진짜값중 가장큰 index값을 준다 
print('예측결과 : ', new_pred)
print(read_model.predict_proba(new_data)) # softmax가 준 진짜값(출력값)


# iris dataset 분류 연습용 시각화 코드
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_regionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')      # 마커 표시 모양 5개 정의
    colors = ('pink', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)   # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)   
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regionFunc(X=x_combined_std, y=y_combined, classifier=read_model, test_idx=range(105, 150), title='scikit-learn제공')  

