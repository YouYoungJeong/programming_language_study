'''
군집분석(Clustering Analysis)
    데이터 간의 유사도를 정의하고 그 유사도에 가까운 것부터 순서대로 합쳐가는 방법으로
    거리나 상관계수등을 이용한다.
    이는 비슷한 특성을 가진 개체를 그룹으로 만들고, 
    군집 분리후 T-test, ANOVA분석 등을 통해 그룹간 평균의 차이를 확인할 수도 있다.

    군집분석은 비지도학습
        데이터만주고 label은 제공하지 않는다.

    Clustering 기법중 계층적 클러스터(HCA)
        - 응집형 : 군집의 크기를 점점 늘리기    (상향식)
        - 분리형 : 군집의 크기를 점점 줄여나가기(하향식)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

np.random.seed(123)
var = ['x','y']
labels = ['점0','점1','점2','점3','점4']
x =  np.random.random_sample([5, 2]) * 10
df = pd.DataFrame(x, columns=var, index=labels)
print(df)

plt.scatter(x[:, 0], x[:, 1], c='magenta', marker='o', s=50)
# text 추가
for i , txt in enumerate(labels):
    plt.text(x[i, 0], x[i, 1], txt)
plt.grid(True)
plt.show()

# 각 점간의 거리 계산
# pdist : 거리 계산
# squareform : 거리계산 사각형으로 결과 확인
from scipy.spatial.distance import pdist, squareform
dist_vec = pdist(df, metric='euclidean')
print("dist_vec :",dist_vec)
print()

# pdist의 결과를 사각형 형식으로 보기
print(squareform(dist_vec))
row_dist = pd.DataFrame(squareform(dist_vec), columns=labels, index=labels)
print(row_dist)
print()

# 응집형 계층적 Clustering
from scipy.cluster.hierarchy import linkage 
row_clusters = linkage(dist_vec, method='ward')
df2 = pd.DataFrame(row_clusters, columns=['cluster_id1', 'cluster_id2', 'distance','cluster_member'])
# 군집 1, 군집2, 두군집과의 거리,  군집멤버
print(df2)
'''
    cluster_id1  cluster_id2  distance  cluster_member
0          0.0          2.0  1.388848             2.0   # 0 ,2
1          4.0          5.0  2.657109             3.0   # 0, 2, 4
2          1.0          6.0  5.454004             4.0   # 0, 1, 2, 4
3          3.0          7.0  6.647102             5.0   # 0, 1, 2, 3, 4
'''

# Cluster의 계층구조를 계통도(dendrogram)으로 출력
from scipy.cluster.hierarchy import dendrogram
row_dene = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('유클리드(euclidean) 거리')
plt.show()