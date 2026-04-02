'''
공분산 / 상관계수
    변수가 하나인 경우에는 분산은 거리와 관련이 있다.
    변수가 두개인 경우에는 분산은 거리와 방향을 갖는다.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 공분산
print(np.cov(np.arange(1, 6), np.arange(2, 7))) # 우상향
print(np.cov(np.arange(10, 60, 10), np.arange(20, 70, 10))) # 우상향
print(np.cov(np.arange(100, 600, 100), np.arange(200, 700, 100))) # 우상향
# 표준화(상관계수)를 하지 않으면 스케일에 따라 값의 차이가 많이남.

print(np.cov(np.arange(1, 6), (3, 3, 3, 3, 3))) # 직선(수평선)
print(np.cov(np.arange(1, 6), np.arange(6, 1, -1))) # 우하향
print()
''' cov의 결과값
[   Var(data1)         Cov(data2,data1)
    Cov(data1,data2)   Var(data2)        ]
'''


x = [8, 3, 6, 6, 9, 4, 3, 9, 3, 4]
print('x평균 : ', np.mean(x))
print('x분산 : ', np.var(x))

y = [6, 2, 4, 6, 9, 5, 1, 8, 4, 5]
print('y평균 : ', np.mean(y))
print('y분산 : ', np.var(y))

# 데이터 시각화하기
# plt.plot(x, y, 'o')
# plt.show()
# 두 변수는 우상향하고 있다. - 관계가 있다.

print('x, y의 공분산',np.cov(x, y))
print('x, y의 공분산',np.cov(x, y)[0, 1]) # 5.222222


# scale up
x2 = [80, 30, 60, 60, 90, 40, 30, 90, 30, 40]
y2 = [6, 2, 4, 6, 9, 5, 1, 8, 4, 5]
print('x2, y2의 공분산',np.cov(x2, y2))
print('x2, y2의 공분산',np.cov(x2, y2)[0, 1]) # 52.222222

# 데이터 시각화하기
# plt.plot(x2, y2, 'o')
# plt.show()
# 데이터의 패턴은 똑같아
print()

"""
상관계수(r)
두 데이터의 단위(scale)에 따라 패턴이 일치할지라도 
공분산의 크기가 달라지므로 절대적 크기 판단이 어려움.
따라서 공분산을 표준화해 -1 0 1 범위로 만든것이 상관계수(r) 이다."""

# 피어슨 상관계수 - 0.3 이상일때만 데이터분석에 참여할 수 있다.
print('x, y의 상관계수 :', np.corrcoef(x, y)) 
print('x, y의 상관계수 :', np.corrcoef(x, y)[0, 1])     # 0.866368
print('x2, y2의 상관계수 :', np.corrcoef(x2, y2)[0, 1]) # 0.866368
print()

# scipy 모듈 사용
print('scipy 모듈 사용(pearson) : ', stats.pearsonr(x, y)) # 피어슨 상관계수
print('scipy 모듈 사용(spearman) : ', stats.spearmanr(x, y))  # 스피어만 상관계수


# 비선형 데이터 - 공분산 상관계수 모두 구하지 못함(측정불가) => 의미없다.
# 데이터는 선형이여야 한다.
m = [-3, -2, -1, -0, 1, 2, 3]
n = [9, 4, 1, 0, 1, 4, 9]
print(np.cov(m, n)[0, 1]) # 0.0
print(np.corrcoef(m, n)[0, 1]) # 0.0
# plt.plot(m, n, 'o')
# plt.show()


