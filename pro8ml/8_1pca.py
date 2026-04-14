'''
주성분 분석(PCA, Principal Component Analysis)
    선형대수 관점에서, 입력데이터의 공분산 행렬을 고유값 분해하고 
    이렇게 구한 고유벡터에 입력데이터를 선형변환하는 것이다.
    이 고유벡터가 PCA의 주성분 벡터로서 입력데이터의 분산이 큰 방향을 나타낸다.
    입력 데이터의 성질을 최대한 유지한 상태로 고차원을 저차원데이터로 변환하는 기법이다.

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
x = iris.data[:n, :2] # sepal weight, height 열만 선택
print('차원 축소 전 :',x, x.shape, type(x)) # (10, 2) <class 'numpy.ndarray'>
print(x.T)

# 시각화
# 패턴이 우하향하고 있기 때문에 패턴이 일정하다 하고 PCA를 적용 할 수 있다.
plt.plot(x.T, 'o:')
plt.xticks(range(2), ['꽃받침길이' ,'꽃받침너비'])
plt.grid(True)
plt.legend(['표본{}'.format(i + 1) for i in range(n)])
plt.title('붓꽃(iris) 크기 특성')
plt.xlabel("특성의 종류")
plt.ylabel("특성값")
plt.xlim(-0.5, 2)
plt.ylim(2.5, 6)
plt.show()