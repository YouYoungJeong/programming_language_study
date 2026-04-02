'''
공분산 / 상관계수
'''
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
np.set_printoptions(suppress=True, precision=10)
# pd.options.display.float_format = '{:.10f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv")
# 데이터 확인
print(data.head())
print(data.describe())

print("표준편차") # 평균과의 거리 확인(방향X)
print(np.std(data.친밀도)) # 0.968505
print(np.std(data.적절성)) # 0.85802
print(np.std(data.만족도)) # 0.82717
print()

# 시각화
# plt.hist([np.std(data.친밀도), np.std(data.적절성), np.std(data.만족도)])
# plt.show()

# 공분산 확인 - 방향O
print("공분산 - numpy(2개 밖에 못본다)")
print(print("공분산 - numpy"))
print(np.cov(data.친밀도, data.적절성))
print(np.cov(data.친밀도, data.만족도))
print()
print("공분산 - pandas(dataframe한번에 확인가능)")
print(data.cov())
print()

# 상관계수
print("상관계수(r) - np")
print(np.corrcoef(data.친밀도, data.적절성))
print(np.corrcoef(data.친밀도, data.만족도))
print()
print("상관계수(r) - pd")
print(data.corr())
print(data.corr(method='pearson'))  # == : 변수가 연속형일때 사용, 정규성을 따름
print(data.corr(method='spearman')) # : 변수가 범주형 - 서열척도 일때, 정규성을 따르지 않음.(비모수검정)
print(data.corr(method='kendall'))
print()

# 만족도에 따른 다른 특성 사이의 상관관계 (정렬보기)
co_re = data.corr()
print(co_re['만족도'].sort_values(ascending=False))
print()

# 시각화하기
data.plot(kind='scatter', x='만족도', y='적절성')
plt.show()

attr = ['친밀도','적절성','만족도']
scatter_matrix(data[attr], figsize=(10,6))
plt.show()

sns.heatmap(data.corr(), annot=True)
plt.show()

# heatmap에 텍스트 표시 추가사항 적용해 보기
corr = data.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)  # 상관계수값 표시
mask[np.triu_indices_from(mask)] = True
# Draw the heatmap with the mask and correct aspect ratio
vmax = np.abs(corr.values[~mask]).max()
fig, ax = plt.subplots()     # Set up the matplotlib figure

sns.heatmap(corr, mask=mask, vmin=-vmax, vmax=vmax, square=True, linecolor="lightgray", linewidths=1, ax=ax)

for i in range(len(corr)):
    ax.text(i + 0.5, len(corr) - (i + 0.5), corr.columns[i], ha="center", va="center", rotation=45)
    for j in range(i + 1, len(corr)):
        s = "{:.3f}".format(corr.values[i, j])
        ax.text(j + 0.5, len(corr) - (i + 0.5), s, ha="center", va="center")
ax.axis("off")
plt.show()