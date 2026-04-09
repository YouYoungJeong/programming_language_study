'''
의사결정나무 (Decision Tree) 분류 모델 - 워밍업
    데이터 균일도에 따른 규칙기반의 결정트리
    트리는 데이터를 직각(수직, 수평) 기준으로 나누면서 영역을 만든다.
'''
from sklearn.datasets import make_classification # 분류용 연습데이터 생성
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np

# 데이터 생성
x , y = make_classification(n_samples=100, n_features=2, n_redundant=0,\
                            n_informative=2, random_state=42) # 의미있는 데이터 2개

# 모델 생성
model = DecisionTreeClassifier(criterion='gini', max_depth=3) 
# criterion기본은 지니지수, max_depth 최대 깊이 3 - 자료구조?: 원하는 모델을 빨리 찾아가는게 있대
model.fit(x, y) # fit(입력데이터, 정답데이터) <-- 지도학습(Supervised Learning)

# 트리구조 시각화
plt.figure(figsize=(10, 6))
# 학습된 결정트리 모델을 시각화함
plot_tree(model, feature_names=['x1','x2'], # 독립변수 이름 지정
                    class_names=['0','1'],  # 종속변수 결과
                            filled=True)    # 노드색깔
# gini = 0 => 불순물이 없다.
# max depth=3 을 줬지만 불순도가 있으면 자기가 알아서 늘어남
plt.show()

# 결정경계 시각화(진짜로 수직수평으로 나누고 있는지)
xx , yy = np.meshgrid( # x축 y축 값을 조합해서 좌표 격자를 만듦.
    # x1범위를 100개의 구간으로 나눔.
    np.linspace(x[:, 0].min(),x[:, 0].max(), 100 ),
    # x2범위를 100개의 구간으로 나눔.
    np.linspace(x[:, 1].min(),x[:, 1].max(), 100 )
)
# 모든 좌표에 대해 예측값 계산
z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # 차원축소(ravel)해서 열추가(c_)
z = z.reshape(xx.shape) # 예측 결과를 원래 grid로 형태 변환
print(z)
# 결정경계 표현 - contour
plt.contour(xx, yy, z, alpha=0.3) # 영역을 색으로 채워 결정경계를 표현한다
plt.scatter(x[:, 0], x[:,1], c=y) # 0아니면 1을 색으로 준다
plt.title("Decision Boundry(결정경계)")
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()