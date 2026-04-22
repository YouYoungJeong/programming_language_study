'''
DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
    머신 러닝에 주로 사용되는 클러스터링 알고리즘으로 Multi Dimension의 데이터를 
    밀도 기반으로 서로 가까운 데이터 포인트를 함께 그룹화하는 알고리즘이다.
    DBSCAN은 밀도가 다양하거나 모양이 불규칙한 클러스터가 있는 데이터와 같이 
    모양이 잘 정의되지 않고 이상치가 많은 데이터를 처리할 때 유용하게 사용 가능하다.
'''
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib import style
from sklearn.datasets import make_moons # 이상치를 만들 수 있는 함수(DBSCAN연습하기 좋다.) 
from sklearn.cluster import DBSCAN, KMeans

# 샘플 데이터 생성
x, y = make_moons(n_samples=200, noise=0.05, shuffle=True, random_state=0)
print(x[:-5], x.shape) # (200, 2)
# print(y)
print()

# 시각화하기
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
# 지도학습인경우 정답을 주기 때문에 어느정도 학습이 되는데 비지도는 어렵다.

# K-means로 군집 분류해보기
print("KMeans로 군집 분류 하기")
km = KMeans(n_clusters=2, init='k-means++', random_state=0)
kmpred = km.fit_predict(x)
print('km 예측군집 id :', kmpred[:10]) # [1 1 0 0 1 1 1 1 1 1]
print()

# km 결과 시각화
def plotResult(x, pr): 
    # Cluster1
    plt.scatter(x[pr==0, 0], x[pr==0, 1], c='darkgreen', marker='o', s=40, label='Cluster1')
    # Cluster 2
    plt.scatter(x[pr==1, 0], x[pr==1, 1], c='purple', marker='s', s=40, label='Cluster2')
    # 중심점
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='red', marker='+', label='Centroid')
    plt.title("Clustering 결과")
    plt.legend()
    plt.show()

plotResult(x, kmpred) # k 설정에 따라 무조건 2개로 분리, 반달을 기준으로 자르기가 되고 있다.

# DBSCAN 으로 군집 분류해보기
print("DBSCAN 으로 군집 분류 하기")
dbscan = DBSCAN(eps=0.2,        # 샘플간 최대 거리, 크기를 조정하면 분류하는 방법이 달라짐.
                min_samples=5,  # 데이터(점)에 대한 이웃 최소 샘플수 : 반경 포인트들의 최소 갯수(유기적으로 조절해야함)
                metric='euclidean'
)
dbpred = dbscan.fit_predict(x)
print("DBSCAN 군집 id :",dbpred[:10])  # [0 1 1 0 1 1 0 1 0 1]
print('군집 종류 :', set(dbpred))      # 0, 1만 있는걸로 보아 이상치는 없는 상태

plotResult(x, dbpred) # 모양에 따라서 군집이 형성됨.
# KMeans는 k개에 따라 군집의 갯수를 맞춤
# DBSCAN은 밀도에 의해 형태를 맞춘다.