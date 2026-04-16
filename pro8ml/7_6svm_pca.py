'''ex40
주성분 분석(PCA, Principal Component Analysis)
    선형대수 관점에서, 입력데이터의 공분산 행렬을 고유값 분해하고 
    이렇게 구한 고유벡터에 입력데이터를 선형변환하는 것이다.
    이 고유벡터가 PCA의 주성분 벡터로서 입력데이터의 분산이 큰 방향을 나타낸다.
    입력 데이터의 성질을 최대한 유지한 상태로 고차원을 저차원데이터로 변환하는 기법이다.
        성격이 비슷한 변수를 합칠 수 있다
        차원축소시 성능이 좋아진다면 안할 이유가 없다 - 속도가 빨라지고, 메모리 사용이 적어짐

PCA는 딥러닝에서도 많이 사용된다 하지만, 딥러닝에서는 PCA를 지원하지 않기 때문에 
전통적인 방법으로 차원을 축소하고 딥러닝으로 넘어가야한다.

    iris dataset 차원축소 
'''
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
n = 10
x = iris.data[:n, :2] # sepal length, weight 열만 선택
print('차원 축소 전 :',x, x.shape, type(x)) # (10, 2) <class 'numpy.ndarray'>
print(x.T)
print()

# 시각화 1 (각 샘플의 두 특성값을 선으로 연결해 비교함)
'''PCA의 근거를 찾기
    패턴이 우하향하고 있기 때문에 패턴이 일정하다 판단
        -> 패턴의 방향이 같다 
        -> PCA를 적용 할 수 있다.
'''
# plt.plot(x.T, 'o:')
# plt.xticks(range(2), ['꽃받침길이' ,'꽃받침너비'])
# plt.grid(True)
# plt.legend(['표본{}'.format(i + 1) for i in range(n)])
# plt.title('붓꽃(iris) 크기 특성')
# plt.xlabel("특성의 종류")
# plt.ylabel("특성값")
# plt.xlim(-0.5, 2)
# plt.ylim(2.5, 6)
# plt.show()


df = pd.DataFrame(x)
print(df)
print()

# 시각화2 - 산점도 그리기(각 샘플의 패턴 확인, 데이터 분포를 확인)
'''PCA의 근거를 찾기
    우상향 하는 패턴을 공통적으로 가지고 있는것을 확인했음. 
        -> 패턴의 방향이 같다 
        -> PCA할 수 있다.
'''
# ax = sns.scatterplot(x=df[0], y=df[1], marker='s', s=100, color='lightgreen') 또는
# ax = sns.scatterplot(x=0, y=1, data=df, marker='s', s=100, color='lightgreen')
# # 각 점에 대해 text 표시
# for i in range(n):
#     ax.text(x[i, 0] - 0.05, x[i, 1] + 0.03, '표본{}'.format(i+1))
# plt.xlabel("꽃받침 길이")
# plt.ylabel("꽃받침 폭")
# plt.title("붓꽃(iris) 특성")
# plt.axis("equal")
# plt.show()

'''
위 두개의 그래프 결과 두 변수는 공통적인 특징이 있으므로 차원축소의 근거가 있다고 판단
    - 상관계수(관계)가 높을 수록 좋다. - 상관관계에 영향이 크다.
    -> PCA를 진행 = 선형변환을 통해 차원을 축소하는것이다.
순서1 : 입력데이터의 공분산 행렬을 생성
순서2 : 공분산 행렬(대칭 행렬)의 고유벡터(방향은 같고 크기만 다름)와 고윳값(고유벡터의 크기)을 계산한다
순서3 : 고윳값이 큰 순서대로 k개(PCA 변환 차수 만큼) 만큼 고유벡터를 추출
순서4 : 고윳값이 가장 큰 순으로 추출된 고유벡터를 이용해 새롭게 입력 데이터를 변환한다.
    sklearn의 PCA를 이용하면 순서대로 진행을 함.
'''
pca1 = PCA(n_components=1)      # 변환할 차원수 (iris 2개의 데이터를 1개로 축소하겠다, 제1~2 주성분을 많이 사용)
x_low = pca1.fit_transform(x)   # 특징 행렬을 낮은 차원의 근사행렬로 변환
print(f"x_low : \n{x_low},\n{x_low.shape}") # (10, 2) => (10, 1)
print()

# 주성분 값 원복하기
# 분류 모델도 100완벽하면 overfitting를 의심해야 한다.
x2 = pca1.inverse_transform(x_low)
print(f"원복 후 : \n{x2},\n{x2.shape}") # (10, 1) => (10, 2)
print()
print('원본 : ', x[0, :])   # [5.1 3.5]
print('PCA : ', x_low[0])   # [0.30270263]
print('원복 : ', x2[0, :])  # [5.06676112 3.53108532]
# print('오차 확인\n',df-x2)
print()

# 시각화 : 주성분 분석값을 기반한 시각화
pc1 = pca1.components_      # 주성분 벡터
print(pc1) # [[0.68305029 0.73037134]]  - PCA 방향 벡터
mean = x.mean(axis=0)     # 데이터 평균 - 중심점
df = pd.DataFrame(x)
# ax = sns.scatterplot(x=0, y=1, data=df, marker='s', s=100, color='magenta') 
# # 각 점에 대해 text 표시
# for i in range(n):
#     ax.text(x[i, 0] - 0.05, x[i, 1] + 0.03, f'표본{i+1}')
# # PCA축 화살표
# plt.quiver(
#     mean[0], mean[1],   # 시작점(평균)
#     pc1[0, 0], pc1[0, 1],     # 방향
#     scale = 3, color='pink', width=0.01
#     )
# plt.xlabel("꽃받침 길이")
# plt.ylabel("꽃받침 폭")
# plt.title("붓꽃(iris) 특성 + 제1 주성분(PCA)")
# plt.axis("equal")
# plt.grid(True)
# plt.show()
# print()

# ====================================================================================
# 원본 열 4개를 차원축소 해 2개의 열로 변환 후  SVM 분류 모델을 작성
print("-"*20," SVM 모델 생성하기 ","-"*20)
x = iris.data
print(x[0,:])           # [5.1 3.5 1.4 0.2]
print(iris.feature_names) 
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
pca2 = PCA(n_components=2)
x_low2 = pca2.fit_transform(x)
print(f'x_low2 :\n{x_low2[0, :]}\nx_low2.shape : {x_low2.shape}') 
# [-2.68412563  0.31939725], (150, 2) : 4열 -> 2열
print()

# 변동성 비율(explained_variance_ratio_)
print(pca2.explained_variance_ratio_) # [0.92461872 0.05306648]

# 원복하기 -> 원복 값이 많이 변하지 않았네 쓸만 하겠다
x4 = pca2.inverse_transform(x_low2)
print('원본 : ', x[0, :])   # [5.1 3.5 1.4 0.2]
print('차원축소(PCA) : ', x_low2[0])   # [-2.68412563  0.31939725]
print('차원복귀(원복) : ', x4[0, :])   # [5.08303897 3.51741393 1.40321372 0.21353169]
print()

# 데이터 준비
iris1 = pd.DataFrame(x, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
print(iris1.head(3))
iris2 = pd.DataFrame(x_low2, columns=['var1', 'var2'])
print(iris2.head(3))
print()

# 데이터 분류
feature1 = iris1.values
print(feature1[:3])
label = iris.target
print(label[:3])
print()

# 원본 데이터 SVM 모델 생성
from sklearn import svm, metrics
print("원본데이터로 SVM분류 모델 작성")
model1 = svm.SVC(C=0.1, random_state=0).fit(feature1, label)
spred1 = model1.predict(feature1)
print("model1 accuracy : ", metrics.accuracy_score(label, spred1)) # 0.94
print()

# PCA데이터 분류
feature2 = iris2.values
print(feature2[:3])

# 주성분(PCA)데이터로 SVM분류 모델 작성
print("주성분(PCA)데이터로 SVM분류 모델 작성")
model2 = svm.SVC(C=0.1, random_state=0).fit(feature2, label)
spred2 = model2.predict(feature2)
print("model2 accuracy : ", metrics.accuracy_score(label, spred2)) # 0.9466
# PCA데이터의 정확도가 약간 높다 이걸 안사용할 이유가 없어. 데이터의 수가 늘어날 수록 중요하다.
print()
