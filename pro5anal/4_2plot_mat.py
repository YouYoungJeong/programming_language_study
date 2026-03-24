import numpy as np
import matplotlib.pyplot as plt
# 차트 영역 객체 선언시 인터페이스 유형에 따른 방식들.

# 1) Matplotlib 스타일의 인터페이스
x = np.arange(10)
plt.figure()
plt.subplot(2, 1, 1) # (row, column, panel number)
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()

# 2) 객체지향 OOP 인터페이스
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.show()

# 차트의 종류 일부 확인
# 히스토그램(hist), plot
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)    # 골격생성
ax2 = fig.add_subplot(1, 2, 2)
ax1.hist(np.random.randn(1000), bins=100, alpha=0.9) # 집계구간(bins)은 10개, 투명도는 0.9다
ax2.plot(np.random.randn(1000))
plt.show()

# 막대그래프(bar chart)
# 세로 막대
data = [50, 80, 100, 90, 70]
plt.bar(range(len(data)), data)
plt.show() 
# 가로 막대
err = np.random.rand(len(data)) # 오차 막대 생성
plt.barh(range(len(data)), data, alpha=0.4, xerr=err)
plt.show() 

# 원 그래프(pie chart) - 많은 양의 데이터를 표시 할때 안 좋음, explode파이 조각 분리
plt.pie(data, colors=['yellow','aqua','magenta','green','orange'], explode=(0, 0.2, 0, 0.1, 0))
plt.title("Pie Chart")
plt.show()

# boxplot : 전체데이터의 분포를 확인하기위해 효과적, 이상치 확인에 도움
# 데이터의 양이 많을때 사용하는 데이터 분포 확인하기 위해 사용
# 이상치 데이터를 확인 하기 위해 사용.
# 박스가운데선은 중앙값.
"""
Boxplot 기준 이상치(outlier)의 정의 ----------------------------------------------------------------
이상치란 Q1 - 1.5 × IQR 보다 작거나, Q3 + 1.5 × IQR 보다 큰 값을 말한다.
Q1: 1사분위수 (하위 25%)
Q3: 3사분위수 (상위 75%)
IQR: Q3 - Q1 (중간 50% 범위)
* 이상치 경계:
    하한선 (lower bound) = Q1 − 1.5 × IQR
    상한선 (upper bound) = Q3 + 1.5 × IQR
    이 두 경계 바깥에 있는 값 → 이상치
"""
data = [1, 50, 80, 100, 90, 70, 300]
plt.boxplot(data)
plt.show()

# (scatter)bubble chart
# 기본은 산포도(scatter) 데이터. scale을 줘야 버블이됨.
# 데이터의 크기를 원의 크기로 나타내는 산점도(산포도)그래프. 
# 점의 크기를 동적으로 표시
np.random.seed(0)
n=30
x = np.random.rand(n)
y = np.random.rand(n)
color = np.random.rand(n)
scale = np.pi * (np.random.rand(n) * 15) ** 2
plt.scatter(x, y, c=color, s=scale)
plt.show()

# 시계열 데이터(시간의 흐름에 따라 만들어지는 데이터)로 선그래프
import pandas as pd
fdata = pd.DataFrame(np.random.randn(1000, 4),
                    index=pd.date_range('1/1/2000', periods=1000), 
                    columns=list('abcd'))
print(fdata.head())
fdata = fdata.cumsum() # 누적합
print(fdata.head())
plt.plot(fdata)
plt.show()

# pandas의 plot기능 사용해보기 - pandas에도 시각화 기능이 있어.
fdata.plot()
fdata.plot(kind='bar')

plt.xlabel('time')
plt.ylabel('data')
plt.show()
