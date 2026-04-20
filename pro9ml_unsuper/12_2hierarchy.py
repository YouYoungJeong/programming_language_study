'''
군집분석(Clustering Analysis) - Clustering 기법중 계층적 클러스터(HCA)
    학생 10명의 시험점수 사용
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

students = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
scores = np.array([76, 95, 65, 85, 60, 92, 55, 88, 83, 72]).reshape(-1, 1)
print('점수 :', scores)

# 계층적 분석
linked = linkage(scores, method='ward')

# 시각화
plt.figure(figsize=(10, 6))
dendrogram(linked, labels=students)
plt.title("Student score hierarchy")
plt.xlabel("Students")
plt.ylabel("Distance")
plt.axhline(y=25, color='pink', linestyle='--', label='cut at 25(t=2)')
plt.axhline(y=15, color='aqua', linestyle='--', label='cut at 15(t=3)')
plt.legend()
plt.grid(True)
plt.show()

# 군집을 3개로 나누기
clusters = fcluster(linked, t=3, criterion='maxclust') # t값 만긐 clust해
print(f'학생별 자료 군집 결과 : ')
for stu, cluster in zip(students, clusters):
    print(f'{stu} : clutser {cluster}')

# 군집별로 점수와 이름 확인
cluster_info = {}
for student, cluster, score in zip(students, clusters, scores.flatten()):
    if cluster not in cluster_info:
        cluster_info[cluster] = {'students':[], "scores":[]}
    cluster_info[cluster]['students'].append(student)
    cluster_info[cluster]['scores'].append(score)
print(cluster_info)
print()

# 군집별 평균점수와 학생 이름 출력
for cluster_id, info in sorted(cluster_info.items()):
    avg_score = np.mean(info['scores'])
    student_list = ", ".join(info['students'])
    print(f'Cluster {cluster_id}의 평균 점수 = {avg_score:.2f}, 학생들 = {student_list}')

# Cluster 1의 평균 점수 = 88.60, 학생들 = s2, s4, s6, s8, s9
# Cluster 2의 평균 점수 = 74.00, 학생들 = s1, s10
# Cluster 3의 평균 점수 = 60.00, 학생들 = s3, s5, s7

# 군집별 산점도
x_positions = np.arange(len(students))
y_socres = scores.ravel()

colors = {1:"blue", 2:"orange", 3:"lightgreen"}
plt.figure(figsize=(10, 6))
for i, (x, y, cluster) in enumerate(zip(x_positions, y_socres, clusters)):
    plt.scatter(x, y, color=colors[cluster], s=100)
    plt.text(x, y + 1.5, students[i], fontsize=12, ha='center')
plt.xticks(x_positions, students)
plt.xlabel("Students")
plt.ylabel("Scores")
plt.title('Score Cluster')
plt.grid(True)
plt.show()

# 성적 그룹을 분석
# 고객 등급 분류, 사용자 행동 패턴 등을 군집화 할 수 있다.

