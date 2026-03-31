'''
단일 모집단의 평균에 대한 가설검정(one samples t-test)
실습 예제 1)
A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 
(국어 점수 평균검정) - student.csv
국어 점수 평균검정(80점이라 가정)
집단 : A중학교 - 단일
귀무가설 : 학샏을의 국어점수 평균값은 80점이다.(모수, 모집단)
대립가설 : 학생들의 국어점수 평균값이 80점이 아니다.
'''
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv")
print(data.head(2),'\n',data.tail(2))
print(data.describe())
print(data['국어'].mean())      # 72.9
print('정원 : ',len(data))      # 20건수
'''
일반적으로 30건이 넘으면 중심극한정리에 의해 정규성을 따른다고 가정한다.
    (30개가 넘으면 정규성 검정을 해도 되고 안해도 됨.)
하지만 데이터가 20건 이므로 모집단의 수가 30이 넘지 않으니 정규성 검정을 해야함
shapiro-wilk test - 정규성 검정 테스트
p값은 alpha = 0.05보다 커야 정규성을 따른다고 할 수 있다.
'''
# 정규성 검정
print(stats.shapiro(data['국어'])) 
# alpha = 0.05 > pvalue=0.01295 : 정규성을 위배함.정규성을 만족하지 못함.


# 정규성을 만족 하지 못한 경우 대안 - Wilcoxon
# Wilcoxon : 비모수 검정 방법으로 정규성이 없을 때 적절한 선택이 될 수 있다.
wilcoxon_result = wilcoxon(data['국어'] - 80) # 이때의 80은 중앙값으로 판단해야한다.
print(wilcoxon_result) # statistic(t)=74.0, pvalue=0.39777
print('Wilcoxon사용시 : alpha = 0.05 < pvalue=0.39777 이므로 귀무가설 채택')

# t-test사용하기.
result = stats.ttest_1samp(data['국어'], popmean=80)
print(result) # statistic=-1.33218, pvalue=0.198560, df=19
print('t-test 사용시 : alpha = 0.05 < pvalue=0.198560 이므로 귀무가설 채택')
'''
원래는 집단이 하나이기 때문에 t-test one sample을 사용
과정은 다를지언정 결과가 비슷하게 나옴.정규성을 만족하지 않더라도  t-test를 사용하는 경우가 많음.
집단이 하나라 정규성에 대한 이야기만 함
두집단이상 일때부터 등분산성 따짐.
'''
print('''
        결론: 정규성은 부족하나 귀무가설 채택이라는 동일 결론을 얻음
        표본 수가 크다면 그냥 ttest_1samp을 써도 된다.

        -보고서 작성시-
        'shapiro-wilk test' 검정 결과 (alpha = 0.05 > pvalue=0.01295) 정규성을 위배하였으나,
        비모수 검정(Wilcoxon) 결과도 동일 하므로  ttest_1samp결과를 신뢰 할 수 있다.라고 명시한다.
    ''')



'''
실습 예제 2)
여아 신생아 몸무게의 평균 검정 수행 babyboom.csv
여아 신생아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자.
이보다 더 크다 : 단측검정 이다


귀무 가설 : 여아 신생아의 몸무게는 평균이 2800(g) 이다.
대립 가설 : 여아 신생아의 몸무게는 평균이 2800(g) 보다 크다
'''
print('-'*20,' 2) one samples t-test 실습','-'*20)
data2 = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/babyboom.csv")
print(data2.head(3))
print(data2.describe())
print()

fdata = data2[data2.gender==1]          # gender 1:여아, 2:남아
print(fdata.head(3),'\n' ,len(fdata))   # 표본개수 18개
print("여아 몸무게 평균 :",np.mean(fdata.weight))       # 3132.44
print("여아 몸무게 표준편차 :",np.std(fdata['weight'])) # 613.787
# 평균 2800 과 3132는 평균에 차이가 있는가?

# one samples t-test 검정하기 (p-value)
result2 = stats.ttest_1samp(fdata['weight'], popmean=2800 )
print("result2 : ", result2) # statistic=2.23318, pvalue=0.0392, df=17
print('''
        해석1 p-value 사용)
        alpha0.05 > pvalue0.0392 이므로 귀무가설 기각
        평균 2800 과 3132는 평균에 차이가 있다라는 의견 수립
        
        해석2 t분포표 사용)
        t값=2.23318, df=17 , alpha=0.05, cv(임계값)=1.740
        t값이 cv(임계치) 오른쪽에 있으므로 (귀무 기각역(영역)에 t값이 위치함).귀무가설 기각
    ''')

print('-'*20,'정규성 검정','-'*20)
# 선행 조건인 정규성 검정을 한 경우
print(stats.shapiro(fdata['weight'])) # pvalue=0.017984
# alpha = 0.05 > pvalue=0.017984 정규성을 만족하지 않음.

# 정규 분포 만족여부 시각화하기 1 (histplot)
sns.histplot(fdata['weight'], kde=True)
plt.show()
# histplot 해석 그래프가 한쪽으로 치우침- 왜곡된 데이터 분포를 확인

# 정규 분포 만족여부 시각화하기 2** (Quantile-Quantile plot)
stats.probplot(fdata['weight'], plot=plt)
plt.show()
'''
Q-Q plot(Quantile-Quantile plot) 해석
데이터를 통과하는 직선 : 회귀(직)선 - 평균을 관통하고 있음.
데이터들이 커브를 그리고 있음. - 좋지 않은 데이터라는 뜻임.
"잔차"가 정규성을 띄워야 하는데 커브를 그리면 정규성을 띌 수 없다.
잔차들이 그래프 선상에 있어야 정규성을 띈다고 함
Q-Q plot상에서 잔차가 정규성을 만족하지 못함.
'''

# 정규성을 만족하지 못하므로 비모수검정인 wilcoxon검정을 실행함
result3 = wilcoxon(fdata['weight']-2800)
print(result3) # statistic=37.0, pvalue=0.034233
print('''
        해석3 wilcoxon 사용)
        alpha0.05 > pvalue=0.034233 이므로 귀무가설 기각
    ''')