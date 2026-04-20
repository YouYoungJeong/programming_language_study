'''
비계층적 군집분석(Non-hierachical Clustering) - K-means
    iris data사용 
    군집분석, 정량평가 , 클러스터별 평균 비교(ANOVA)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
# adjusted_rand_score : 군집 VS 실제 라벨 비교
# normalized_mutual_info_score : 정보량 기반 유사도(같은 정보 공유)
# silhouette_score : 군집 자체 품질 평가(군집에 잘 속해 있는가 확인)
from sklearn.decomposition import PCA # x, y 두축만 사용하기위해 4 -> 2차원 축소
# kmeans - Warning 무시
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")


iris = load_iris()
x = iris.data
y = iris.target
featuer_names = iris.feature_names

df = pd.DataFrame(x, columns=featuer_names)
print('iris data shape', x.shape) # (150, 4)
print()

# 표준화(StadardScaling)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled[:2]) # [[-0.90068117  1.01900435 -1.34022653 -1.3154443 ]...
print()

# KMeans 모델(원본 기반)
# PCA데이터는 정확도가 떨어졌기 때문에 그냥 원본 데이터 사용
# 차원이 많은 경우에는 PCA사용하는걸 권장
k = 3
kmeans = KMeans(
    n_clusters=k, init='k-means++', n_init=10, random_state=42
) 
clusters = kmeans.fit_predict(x_scaled)
df['cluster'] = clusters
print('Cluster 중심값 :',kmeans.cluster_centers_)

# PCA 차원축소 -> 4차원 데이터를 2차원 데이터로 줄이는 이유는 시각화를 위해
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
print(x_pca[:2]) # [[-2.26470281  0.4800266 ].. : 2차원으로 변경함
print(f'pca 설명 분산 비율 값 : {pca.explained_variance_ratio_}')  
print(f'pca 설명 분산 비율 합친 값 : {pca.explained_variance_ratio_.sum():2%}')  
# [0.72962445 0.22850762] 
# -> 다합치면 95.813207% 원본에서 주성분으로 바뀌었을때 원본을 95%정도 설명할 수 있다.
# -> 5%는 설명 할 수 없을 수 있다.
# -> 제1주성분, 제2주성분의 각 주성분이 원본 분산을 얼마나 설명하는지

# 시각화(PCA기반 - 4개의 열을 2차원으로 차트에 표현불가. 따라서 차원축소가 필요함)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=clusters, palette='Set1')
plt.title("KMeams Clustering")
plt.xlabel("PC1(제1 주성분)")
plt.ylabel("PC2(제2 주성분)")
plt.show()

# 실제 라벨과 군집 비교(교차표) - 설명하는것이 중요하다.
ct = pd.crosstab(y, clusters)
print(ct)
# col_0   0   1   2     <- 열 : 군집 번호(Kmeans결과)
# row_0                
# 0       0  50   0     <- setosa
# 1      39   0  11     <- versicolor(?) 섞임 - 거리기반으로 나뉨
# 2      14   0  36     <- versinica(?) 섞임 - 거리기반으로 나뉨
# 행: 실제 라벨(iris)
print('클래프별 대표 군집')
for i in range(ct.shape[0]):
    max_cluster = ct.iloc[i].idxmax()
    print(f'실제 클래스 {i} -> 군집 {max_cluster}')
# 실제 클래스 0 -> 군집 1
# 실제 클래스 1 -> 군집 0
# 실제 클래스 2 -> 군집 2
print()

# 정량 평가하기(ARI, NMI)
print('정량 평가')
ari = adjusted_rand_score(y, clusters)
nmi = normalized_mutual_info_score(y, clusters)
sil_score = silhouette_score(x_scaled, clusters)
print(f'ARI : {ari:.4f}')                   # ARI : 0.6201
print(f'MNI : {nmi:.4f}')                   # MNI : 0.6595
print(f'silhoutte_score : {sil_score:.4f}') 
# silhoutte_score : 0.4599 - 군집 자체 품질평가
# 결과 확인 : 1에 근사 할 수록 좋음. 0 또는 음수면 잘못된 군집
# 좋은 군집이란? 군집내 요소끼리는 가깝고 다른 군집 간에는 거리가 멀어야 한다. 
print()

# k=3을 사용했는데 과연 3이 합리적인지 확인
# 엘보우로 확인
initia_list = []
k_range = range(1, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit_predict(x_scaled)
    initia_list.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_range, initia_list, marker='o')
plt.title("엘보우 기법")
plt.xlabel("클러스터 수(k)")
plt.ylabel("initia")
plt.show()
# k가 가장 완만해 지는 시기가 k=3이므로 3이 가장 적당하다.

# 실제 VS 군집 비교 시각화
plt.figure(figsize=(12, 5))

# 실제 라벨
plt.subplot(1, 2, 1)
sns.scatterplot(x=x_pca[:,0], y=x_scaled[:,1], hue=y, palette='Set1')
plt.title("실제 라벨")

# 군집 결과
plt.subplot(1, 2, 2)
sns.scatterplot(x=x_pca[:,0], y=x_scaled[:,1], hue=clusters, palette='Set1')
plt.title("군집 결과")
plt.show()

# 클러스터 평균 분석 - 가설검정
clusters_mean = df.groupby('cluster').mean()
print("클러스터별 평균 :", clusters_mean)

# 군집 3개 : 군집간 평균 차이 검정 (ANOVA)
# 귀무 : 군집간 평균의 차이가 없다.   
# 대립 : 군집간 평균의 차이가 있다.
from scipy.stats import f_oneway

# 각 군집별 데이터 분리
for col in featuer_names:
    group0 = df[df['cluster']==0][col]
    group1 = df[df['cluster']==1][col]
    group2 = df[df['cluster']==2][col]

    # ANOVA 수행
    f_stat, p_val = f_oneway(group0, group1, group2)
    print(f'{col} : f-statistic : {f_stat:.4f}, p-value : {p_val:.4f}')

    # 해석
    if p_val >= 0.05:
        print("군집간 평균 차이가 없다.(유의하지 않다,우연이다:대립 기각)")
    else:
        print("군집간 평균 차이가 있다.(유의하다,우연이 아니다:귀무 기각)")
    print()
# 군집 품질평가 지표(ARI, NMI, Silhouette)상 군집화 성능이 어느 정도 양호하게 나타났음
# ANOVA에서도 각 변수의 군집 간 평균 차이가 유의하게 확인되었음
# 따라서 해당 군집은 실제 데이터 구조를 어느 정도 반영하는 의미 있는 군집으로 해석할 수 있음

# 사후검정(원래는 모든 칼럼 다 진행해야한다)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# petal lenght로 작업
featuer = 'petal length (cm)'
tukey = pairwise_tukeyhsd(
    endog=df[featuer], groups=df['cluster'], alpha=0.05
)
print('tukeyhsd 결과(petal length) :',tukey)
# ===================================================
# group1 group2 meandiff p-adj  lower   upper  reject
# ---------------------------------------------------
#      0      1  -2.9078   0.0 -3.1405 -2.6751   True
#      0      2   1.1408   0.0  0.9043  1.3773   True
#      1      2   4.0486   0.0  3.8088  4.2884   True
# ---------------------------------------------------
print()

# 사후검정 시각화
tukey.plot_simultaneous(figsize=(6,4))
plt.title(f"tukeyhsd - {featuer}")
plt.xlabel("평균 차이")
plt.show()
print()

# 군집별 boxplot
for col in featuer_names:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x='cluster',y=col, data=df)
    plt.title(f"{col} by cluster")
    plt.show()

# 클러스터 평균 분석 마지막 열에 Type추가(군집별 이름주기)
clusters_mean['label'] = ['Type A', 'Type B', 'Type C']
print(clusters_mean)