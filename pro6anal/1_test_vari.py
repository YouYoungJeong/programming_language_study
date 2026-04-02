"""
표준편차, 분산의 중요성
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

np.random.seed(42)
target_mean = 60
std_dev_small = 10
std_dev_large = 20

print('-'*15 ,' sampling data 생성 ','-'*15)
# sampling data 생성
class1_raw = np.random.normal(loc=target_mean, scale=std_dev_small, size=100)
class2_raw = np.random.normal(loc=target_mean, scale=std_dev_large, size=100)

print(class1_raw[:5])
print()

print('-'*15 ,' 평균 수치 보정하기 ','-'*15)
# 평균수치 보정하기
class1_adj = class1_raw - np.mean(class1_raw) + target_mean
print(class1_adj[:5])
class2_adj = class2_raw - np.mean(class2_raw) + target_mean
print(class2_adj[:5])
print()

print('-'*15 ,' 정수화 하기 + np.clip() ','-'*15)
# 정수화 하기
'''
np.clip(배열, 최소값, 최대값) 
=> 배열값(5, 45, 78)에 
np.clip(배열데이터, 10, 60) -> 10보다 작은값은 10으로 묶이고, 60보다 큰값은 60으로 묶임
지정한 범위 안으로 집어넣음.
'''
class1 = np.clip(np.round(class1_adj), 10 , 100).astype(int)
print(class1[:10])
class2 = np.clip(np.round(class2_adj), 10 , 100).astype(int)
print(class2[:10])
print()

print('-'*15 ,' 기술통계 계산(평균, 표준편차, 분산) ','-'*15)
# 기술통계 계산
#평균
mean1, mean2 = np.mean(class1), np.mean(class2)
# 표준편차
std1, std2 = np.std(class1), np.std(class2)
# 분산
var1, var2 = np.var(class1), np.var(class2)
print("1반(성적 편차가 작음)")
print(f"평균 : {mean1}, 표준편차 : {std1}, 분산 : {var1}")
print("2반(성적 편차가 큼)")
print(f"평균 : {mean2}, 표준편차 : {std2}, 분산 : {var2}")
print()

print('-'*15 ,' DataFram으로 만든후 csv파일로 저장 ','-'*15)
# 파일로 저장
df = pd.DataFrame({
    'class' : ['1반'] * 100 + ['2반'] * 100,
    'score' : np.concatenate([class1, class2])
})
print(df.head(),"\n",df.tail())
df.to_csv('test1vari.csv', index=False, encoding='utf-8-sig')
print()

print('-'*15 ,' 시각화하기 ','-'*15)
# 시각화 하기
# 산포도
x1 = np.random.normal(1, 0.05, size=100)
x2 = np.random.normal(2, 0.05, size=100)
plt.figure(figsize=(10, 6))
plt.scatter(x1, class1, alpha=0.8, label=f'1반(평균={mean1:2f}, 표준편차={std1:2f})')
plt.scatter(x2, class2, alpha=0.8, label=f'2반(평균={mean2:2f}, 표준편차={std2:2f})')
plt.hlines(target_mean, 0.5, 2.5, colors='red', \
            linestyles='dashed',label=f'공통평균={target_mean}')
plt.xticks([1, 2],['1반','2반'])
plt.ylabel("시험 점수")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# boxplot
plt.figure(figsize=(8, 5))
plt.boxplot([class1,class2], label=['1반','2반'])
plt.grid(True)
plt.show()

# histogram
plt.figure(figsize=(10, 6))
plt.hist(class1, bins=30, alpha=0.6 ,label='1반', edgecolor='black')
plt.hist(class2, bins=30, alpha=0.6 ,label='2반', edgecolor='blue')
plt.axvline(target_mean, color='red', linestyle='dotted', label=f'공통평균={target_mean}')
plt.xlabel('시험점수')
plt.ylabel('빈도')
plt.legend()
plt.tight_layout()
plt.show()

"""EDA
    국어 선생 입장 : 두반의 국어 점수의 표준편차(평균)는 차이가 없게 내겠다
    실험 결과 : 차이가 있다. 라고 주장. 
    국어선생님 입장은 classic하게 생각한다면 - 하나의 가설 - 귀무가설
    귀무가설 : 전통적인 주장
    대립가설:  실험을 통해 데이터 수집 후 두반의 점수에 대한 통계 계산후 새로운 주장(의견)을함
    가설검정 t-test를 통해 결정해줌 이때 사용하는 기준값이 p-value
    두집단의 평균을 이용하면 t-test
    가설 검정(t-test)을 통해 두 의견의 채택, 기각을 판단 할 수 있다.
    
"""