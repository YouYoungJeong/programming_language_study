'''
군집분석(Clustering Analysis) - Clustering 기법중 계층적 클러스터(HCA)
    데이터를 단계적으로 묶어 군집(Cluster)를 형성하는 알고리즘
    거리가 가까운 데이터부터 계속 묶어가는 방식
    군집 수를 미리 정하지 않아도 됨. (비계층적을 하기 위해) 구조는 덴드로그램으로 확인
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster # sklearn보다 scipy가 훨씬 좋기 때문에 이걸 사용

iris = load_iris()
x = iris.data
y = iris.target
labels = iris.target_names


df  = pd.DataFrame(x, columns=iris.feature_names)
print(df.head(3))

# Scaleing
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# hierarchy
z = linkage(x_scaled, method='ward')

# dendrogram
plt.figure(figsize=(12, 5))
dendrogram(z)
plt.title("아이리스로 계층적 군집")
plt.xlabel("샘플 데이터")
plt.ylabel("유클리드 거리")
plt.show()

# 댄드로그램을 잘라서 최대  3개의 군집 만들기
clusters = fcluster(Z=z, t=3, criterion='maxclust')
df['cluster'] = clusters
print(df.head())

# 2개 feature 시각화(산점도 사용)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=x_scaled[:,0], y=x_scaled[:,1], hue=clusters, palette='Set1')
# hue=clusters : 군집 결과에 따라 색을 달리 표시
# palette= 색상 스타일 지정
plt.title("Cluster 결과")
plt.xlabel("featuer1")
plt.ylabel("featuer2")
plt.show()      # 꽤 비슷하게 나눠짐. 군집은 정답 label이 없음.

print("실제 라벨 : ",y[:10])        # [0 0 0 0 0 0 0 0 0 0]
print("군집 결과 : ",clusters[:10]) # [1 1 1 1 1 1 1 1 1 1] 실제 0이 군집1로 군집화됨

# 군집 결과 검증하기
print("군집결과 검증")
print("교차표 - 실제 라벨 vs 군집 결과")
ct = pd.crosstab(y, clusters)
print(ct)
# col_0   1   2   3
# row_0            
# 0      49   1   0     <- 잘 분류
# 1       0  27  23     <- versicolor : 많이 섞임(경계가 애매)
# 2       0   2  48     <- 잘 분류
# row_0 (실제 라벨) : 0 -stosa, 1-versicolor, 2-versinica
# col_0 (군집 라벨) : 1 -cluster1, 2-cluster2, 3-cluster3
# setosa는 완벽히 분리되었고, versicolor와 virginica는 일부 섞인 결과 보임
print()

print("교차표 보조 설명 : 각 실제 클래스가 가장 많이 속한 군집")
for i in range(ct.shape[0]):
    max_cluster = ct.iloc[i].idxmax()
    print(f'실제 클래스 {i} → 군집 {max_cluster} (갯수 : {ct.iloc[i].max()})')
# 실제 클래스 0 → 군집 1 (갯수 : 49)
# 실제 클래스 1 → 군집 2 (갯수 : 27)
# 실제 클래스 2 → 군집 3 (갯수 : 48)
print()

# 정량적 평가 : 군집 결과과 실제 정답과 얼마나 유사한지를 수치로 표현
from sklearn.metrics import adjusted_mutual_info_score 
# ARI(Adjusted Rand Index) : 같은 그룹끼리 잘 묶였는지 평가 
# 해석기준 : 0.7 이상 (매우 잘된 그룹), 0.5 ~ 0.7(잘된 그룹(mornal)), 0이하(문제 있음.)
# - 보고서 만들때 사용하면 좋다.
ari = adjusted_mutual_info_score(y, clusters)
print(f'평가 지표 : ARI = {ari:.4f}') # ARI = 0.6713

from sklearn.metrics import normalized_mutual_info_score
# NMI(Normalized Mutual Imfo) - 정보량 기준 얼마나 유사한지 확인 하는 방법
# (그룹간 얼마나 같은 정보를 공유하는가)
# 해석기준 - 0~1사이 이며 ,1 :완벽, 0: 완전 다름
nmi = normalized_mutual_info_score(y, clusters)
print(f'평가 지표 : NMI = {nmi:.4f}') # NMI = 0.6755 - 정보량이 어느정도 공유가 되고 있다.
print()
