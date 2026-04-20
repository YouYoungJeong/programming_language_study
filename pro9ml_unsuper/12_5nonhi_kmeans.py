'''
비계층적 군집분석(Non-hierachical Clustering) - K-means

    계층적군집분석과 같은 학생 10명의 시험점수 사용
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.cluster import KMeans

students = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
scores = np.array([76, 95, 65, 85, 60, 92, 55, 88, 83, 72]).reshape(-1, 1)
# print('점수 :', scores)

# k = 3
kmeans = KMeans(n_clusters=3, init="k-means++" ,random_state=0) # init은 k-means++가 defualt
km_clutsers = kmeans.fit_predict(scores)
print(km_clutsers) # [2 0 1 2 1 0 1 0 2 2]

df = pd.DataFrame({
    "student"   : students,
    'score'     : scores.ravel(),
    'cluster'   : km_clutsers
    })
print(df)
print()

# 군집별 평균점수 구하기
print("군집별 평균점수 출력하기")
grouped = df.groupby('cluster')['score'].mean()
print(grouped)
# 0    91.666667
# 1    60.000000
# 2    79.000000
print()

# 시각화하기(산점도)
x_position = np.arange(len(students))
y_scores = scores.ravel()
colors={0:'pink',1:'indigo',2:'lightgreen'}
plt.figure(figsize=(10, 6))

# 학생별 군집 색으로 구분해 산점도 출력
for i, (x,y,clutsers) in enumerate(zip(x_position,y_scores,km_clutsers)):
    plt.scatter(x, y, c=colors[clutsers], s=100)
    plt.text(x, y + 1.5, students[i], fontsize=10, ha='center')

# 중심점
centers = kmeans.cluster_centers_
for center in centers:
    plt.scatter(len(students)//2, center[0], marker='X', c='red', s=200)

plt.xticks(x_position, students)
plt.xlabel("학생명")
plt.ylabel("점수")
plt.grid(True)
plt.show()