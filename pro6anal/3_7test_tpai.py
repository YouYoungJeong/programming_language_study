'''
paired samples t-test

실습) 복부 수술 전 9명의 몸무게와 복부 수술 후 몸무게 변화
baseline = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
follow_up = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]

귀무가설 : 복부 수술 후 몸무게의 변화는 없다.
대립가설 : 복부 수술 후 몸무게의 변화는 있다.
'''
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib

baseline = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
follow_up = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]

# 데이터 확인하기
print('-'*20,' 데이터 확인하기 ','-'*20)
print(np.mean(baseline))
print(np.mean(follow_up))
print(f'평균의 차이 : {(np.mean(baseline)-np.mean(follow_up)):.2f}') # 6.91
plt.bar(np.arange(2), [np.mean(baseline), np.mean(follow_up)])
plt.xlim(0, 1)
plt.xlabel('수술 전후', fontdict={'fontsize':'12','fontweight':'bold'})
plt.show()
print()

# 검정
print('-'*20,'paired samples t-test','-'*20)
result = stats.ttest_rel(baseline, follow_up)
print(result)
# statistic=3.668116, pvalue=0.0063266, df=8
print('''
    해석)
    유의확률(0.05) > pvalue=0.0063266 이므로 귀무가설 기각.
    복부 수술 후 몸무게의 변화는 있다.라는 주장을 받아들임.
''')
print()


print('-'*20,' 정규성 검정 ','-'*20)
print(stats.shapiro(baseline).pvalue)
print(stats.shapiro(follow_up).pvalue)
