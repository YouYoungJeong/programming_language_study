# Embedding Data로 Clustering 후 시각화
# 서로 비슷한 문장끼리 그룹화

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

texts = [
    '나는 사과를 좋아해',
    '바나나는 내가 제일 좋아하는 과일이야',
    '파이썬은 프로그래밍 언어',
    '나는 가끔 파이썬으로 소스를 만들어',
    '사과와 바나나는 모두 맛있어',
    '파이썬 코딩은 즐거워',
    '나는 망고 스무디를 즐겨 마셔',
    '과일은 건강한 간식이야',
    '나는 열대 과일이 좋아',
    '운동은 역시 축구야',
    '재미있는 야구 경기를 보러 가야지',
    '야구 만세'
]


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
print(embeddings[:3])
print('\n')


#######################################################################################
# KMeans Clustering : K 값 ?
# Cluster 수 찾기 - silhouette 기법
from sklearn.metrics import silhouette_score
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    print(f'k = {k}, score = {score:.4f}')

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)
# print(labels)

print('유사도 기반 문장 클러스터링 결과 : ')
for idx, (text, label) in enumerate(zip(texts, labels)):
    print(f'[Cluster {label}] {text}')
print('\n')


print('<군집 결과>')
from collections import defaultdict
clusters = defaultdict(list)
for text, label in zip(texts, labels):
    clusters[label].append(text)

for cluster_id, docs in clusters.items():
    print(f'\n---cluster {cluster_id}---')
    for d in docs:
        print(d)
# ⇨ 한글이어서 결과가 이상할 수 있음


#######################################################################################
# 군집 결과 시각화
# PCA를 이용해, 차원 축소 후 시각화 (384차원 → 2차원)
# 각 클러스터별 대표 문장 출력

import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)     # 384 → 2

plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'orange', 'purple']
for i in range(n_clusters):
    cluster_points = reduced[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i % len(colors)], label=f'Cluster {i}')

plt.title('문장 군집화(PCA 시각)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.tight_layout()
plt.show()


#######################################################################################
# 군집별 대표 문장 추출
from sklearn.metrics import pairwise_distances_argmin
for i in range(n_clusters):
    cluster_indices = np.where(labels == i)[0]
    cluster_embeddings = embeddings[cluster_indices]    # 특정 군집에 속한 벡터들만 골라내는 역할

    center = kmeans.cluster_centers_[i].reshape(1, -1)
    closet_idx = pairwise_distances_argmin(center, cluster_embeddings)
    closet_text = texts[cluster_indices[closet_idx[0]]]
    print(f'[군집{i} 대표문은 {closet_text}]')