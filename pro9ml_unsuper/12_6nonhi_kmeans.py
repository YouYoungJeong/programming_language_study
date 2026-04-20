'''
비계층적 군집분석(Non-hierachical Clustering) - K-means
    쇼핑몰 고객세분화를 위해 Clustering을 진행
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.cluster import KMeans

# 가상의 고객 데이터
np.random.seed(0)
n_customers = 200 # 고객 수
annual_spanding = np.random.normal(50000, 15000, n_customers) # 연간 지출액(평균, 표준편차, 갯수)
monthly_visits =np.random.normal(5, 2, n_customers) # 월 방문 횟수

# 구간나누기(음수 제거 - clip사용해 음수는 0으로 대체)
# a = np.array([-3.2, -0.5, 1.7]) => np.clip(a, 0, 1) -> [0, 0, 1]
annual_spanding = np.clip(annual_spanding, 0, None)
monthly_visits = np.clip(monthly_visits, 0, None)

data = pd.DataFrame({
    'annual spanding' : annual_spanding,
    'monthly visits' : monthly_visits
})
print(data.head())
print(data.shape) # (200, 2)
print()

# 데이터 확인을 위한 시각화(산포도)
plt.scatter(data['annual spanding'], data['monthly visits'], marker='v', c='indigo')
plt.xlabel("연간 지출(소비) 액")
plt.ylabel("한달 방문수")
plt.title("소비자 분포 산점도")
plt.grid(True)
plt.show()

# 엘보우, 실루엣 기법 사용
# pass

# KMeans 군집화
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(data)

data['cluster'] = clusters
# print(data.head())
# 군집 결과 시각화
for cluster_id in np.unique(clusters):
    # cluster 별로 데이터 추출
    cluster_data = data[data['cluster'] == cluster_id]
    
    # cluster 별로 데이터 보기
    print(data[data['cluster'] == cluster_id])
    
    # cluster 별로 데이터 산점도 찍기 
    plt.scatter(cluster_data['annual spanding'], cluster_data['monthly visits'], label=f'cluster(군집) {cluster_id}')

# 중심점 표시
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
            c='indigo', marker='X', s=200, label='중심점')

plt.xlabel('연간 지출')
plt.ylabel("한달 방문수")
plt.title("소비자 군집 현황 산점도")
plt.legend()
plt.grid(True)
plt.show()
