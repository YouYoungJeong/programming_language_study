'''
independent two samples t-test : 독립표본 t검정
서로 독립인 두 집단의 평균에 대한 통계 검정
비교를 위해 평균과, 표준편차 통계량을 사용함.
두 집단의 평균과 표준편차 비율에 대한 재조 검정법이다.


두 집단의 가설검정 실습 시 분산을 알지 못하는 것으로 한정하겠다
남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 
서로 독립인 두 집단에서 얻은 표본을 독립표본(two sample)이라고 한다.

실습) 남녀 두 집단 간 파이썬 시험의 평균 차이 검정
    남녀의 시험 평균이 우연히 같을 확률은 얼마나 될까?
    만약 우연히 발생 했다면 평균은 같은 것이고, 우연이 아니면 평균은 다른 것 이다.
    95% 신뢰구간에서 우연히 발생할 확률이 5%이상이면 귀무가설 채택

    귀무가설 : 남녀 두 집단 간 파이썬 시험의 평균 차이는 없다
    대립가설 : 남녀 두 집단 간 파이썬 시험의 평균 차이는 있다
'''

from scipy import stats
import pandas as pd
import numpy as np

male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]

print(np.mean(male), np.mean(female)) # 83.8 72.24


# 독립표본 t검정
print('-'*20,' 독립표본 t검정 ','-'*20)
two_sample = stats.ttest_ind(male, female) # 두개의 표본에 대한 독립표본 t검정 수행
tv , pv = two_sample
print("t 검정 통계량 :",tv) # 1.233193127514512
print("p-value :",pv)       # 0.2525076844853278
# df=8 
print('''
    해석) p-value 0.25250 > 0.05(α) 이므로 귀무 채택, 통계적으로 유의하지 않다.
        남녀 두 집단 간 파이썬 시험의 평균 차이는 없다
    ''')

'''
선행조건 1) 두집단이 각각 정규분포를 따라야 한다.
    - 만약 두집단의 표본 수가 30개 이상인경우 정규분포를 따른다고 가정함.(중심극한의 원리) 
        ->  정규성 검정을 안해도 됨
    - 만약 정규성 검정을 했는데 정규성을 만족하지 못할경우 비모수검정 Mann-whitney검정을 한다.
        ->  stats.mannwhitneyu(group1, group2) 
            p > 0.05 인 경우 차이 증거 없음.
            아니면 두집단 평균 차이 있음.
'''
print('-'*20,' 선행조건 1) 정규성 검정','-'*20)
print('male 정규성 검정 : ',stats.shapiro(male))       # 0.6003714029870378
print('female 정규성 검정 : ',stats.shapiro(female))   # 0.778043110871599
print('두집단 정규성을 만족함, 둘 중에 하나만 만족하는경우도 만족했다고 가정함.')
print()

'''
선행조건 2) 두집단의 분산이 같다는 가정이 필요하다.
    - 등(等)분산성 : 데이터의 퍼짐 정도
    - 모듈 : from scipy.stats import levene, bartlett, fligner
    - levene을 가장 많이 사용함.
                        정규성            |       이상치
    -> levene      상관없이 사용 가능     |   이상치에 민감,      =>어떤 분포도 사용 가능
    -> bartlett    만족 했을 때만 가능    |  이상치에 덜 민감,   =>정규분포 일때만 사용

    - p-value 값 >= 0.05 유의수준보다 컸을때 등분산성을 만족한다고 할 수 있다.
    - 등분산성을 만족하는 경우 사용 ▼
        -> stats.ttest_ind(group1, group2,equal_var=True )
    - 등분산성을 만족하지 못하는 경우 ▼ 
        -> Welch's t-test (equal_var=False만 해주면 됨.)
        -> stats.ttest_ind(group1, group2,equal_var=False )
'''
from scipy.stats import levene, bartlett, fligner
print('-'*20,' 선행조건 2) 등분산성 검정','-'*20)
levene_stat, levene_p = levene(male, female)
print("levene_stat :",levene_stat)
print("levene_p : ", levene_p) # 0.49565 >= 0.05 등분산성 만족
# 등분산성을 만족하는 경우 사용 ▼
two_sample = stats.ttest_ind(male, female)   # 아래와 동일 , 두 집단의 분산이 같은 경우로 가정
two_sample = stats.ttest_ind(male, female, equal_var=True)


