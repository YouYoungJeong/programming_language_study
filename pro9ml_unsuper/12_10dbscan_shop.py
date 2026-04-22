'''
DBSCAN(Density-Based Spatial Clustering of Applications with Noise)

    어느 쇼핑몰에 고객 행동 데이터를 이용해 군집 분류 (가공된 데이터 사용)
    고객마다 소비 패턴이 다르므로 여러 그룹이 형성이 됨.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
# 일반적으로 계층적/비계층적 군집 분석을 선행하고, 마음에 안들면 DBSCAN을 함.

np.random.seed(42)

# 데이터 생성(의도적으로 만듦- 데이터 수집이 어려운 관계로)
# vip 고객
vip = pd.DataFrame({
    'annual_spending':np.random.normal(700, 40, 80),
    'visit_per_month':np.random.normal(20, 2, 80),
    'avg_purchase':np.random.normal(80, 10, 80),
    'group':'vip'
})
# 일반 고객 - 평균적인 소비 패턴
normal = pd.DataFrame({
    'annual_spending':np.random.normal(300, 100, 150),
    'visit_per_month':np.random.normal(10, 4, 150),
    'avg_purchase':np.random.normal(30, 15, 150),
    'group':'normal'
})
# 저활동 고객 - 방문 적음, 구매 적음
low = pd.DataFrame({
    'annual_spending':np.random.normal(100, 30, 70),
    'visit_per_month':np.random.normal(3, 1, 70),
    'avg_purchase':np.random.normal(10, 5, 70),
    'group':'low'
})

print(low.head(2))

# 특이 패턴 고객(비선형 패턴) 생성 - 일정하지 않은 소비 패턴 
t = np.linspace(0, 3 * np.pi, 60)
curve = pd.DataFrame({
    'annual_spending':np.random.normal(0, 10, len(t)) + 200 + 100 * np.cos(t),
    'visit_per_month':np.random.normal(0, 1, len(t)) + 10 + 5 * np.sin(t),
    'avg_purchase':40 + 10 * np.sin(t),
    'group':'curve'
})

# 이상 고객 (이상치) - 너무 많이 사거나 거의 안사거나
outliers = pd.DataFrame({
    'annual_spending':[900, 50, 850],
    'visit_per_month':[10, 1, 25],
    'avg_purchase':[120, 5, 100],
    'group':'outlier'
})

# 데이터 병합
df = pd.concat([vip, normal, low, curve, outliers], ignore_index=True)
print(df.head(2))
print()

# 원본데이터(초기데이터) 시각화
plt.figure(figsize=(6,5))
sns.scatterplot(
    x = df['annual_spending'],
    y=df['visit_per_month'],
    hue=df['group'], # 구분기준
    palette='Set2'
)
plt.title("원본(초기) 데이터")
plt.xlabel("연간 소비")
plt.ylabel("월 방문 수")
plt.legend(title='소비 패턴')
plt.show()

# DBSCAN
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df.drop(columns=['group']))
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
clusters = dbscan.fit_predict(x_scaled)
df['cluster'] = clusters
print(df.head(2))

# 군집 결과 시각화
plt.figure(figsize=(6,5))
sns.scatterplot(
    x = df['annual_spending'],
    y=df['visit_per_month'],
    hue=df['cluster'], # 구분기준
    palette='Set1'
)
plt.title("군집 결과 데이터")
plt.xlabel("연간 소비")
plt.ylabel("월 방문 수")
plt.legend(title='소비 패턴')
plt.show()
# 매출에 따라 소비행태에 따라 3개의 군집으로 분류함
# -1 은 이상치

print()
print(f'각 군집 평균 \n\
{df.groupby('cluster')[['annual_spending','visit_per_month','avg_purchase']].mean()}')
# 각 군집 평균 
#          annual_spending  visit_per_month  avg_purchase
# cluster                                                
# -1            445.483569        13.956698     46.090559
#  0            695.876369        19.989732     80.808478
#  1            224.488927         8.087148     27.836348