"""
집단 간 차이분석: 평균 또는 비율 차이를 분석
모집단에서 추출한 표본정보를 이용하여 모집단의 다양한 특성을 과학적으로 추론할 수 있다.

단일 표본 t검정(one samples t-test)
정규분포의 표본에 대한 기대값을 조사하는 검정방법
예상 평균값과 표본 자료간의 평균차이를 검정
독립변수(x) : 범주형
종속변수(y) : 연속형
하나의 집단에  대한 표본 평균이 예측된 평균(모집단)과 같은지 여부를 확인
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

'''
실습 1 : 어느 남성 집단의 평균 키 검정
귀무 가설(H₀) : 해당 집단의 평균 키가 177이다.(모수 - 모집단)
대립 가설(H₁) : 해당 집단의 평균 키가 177이 아니다. - 클 수도 작을 수도 있다
                (=표본데이터의 평균이 177이 아니다)
'''
print('-'*20,' 1) one samples t-test 실습','-'*20)
one_sample = [167.0, 182.7, 169.6, 176.8, 185.0]
print(np.array(one_sample).mean()) # 176.2199

# 차이가 있냐 없냐에 대한 검정
result = stats.ttest_1samp(one_sample, popmean=177) # (데이터, popmean - 예상평균값(모수의 평균))
print(result) # t=-0.2213, pvalue=0.8356, df=4
# 해석
print(''' 
    유의수준0.05 < pvalue0.8356 이므로 귀무채택 
    이 데이터는 우연히 발견된 데이터이다. 따라서 해당 집단의 평균 키가 177이다.라는 가설 유지
    \n''')

# 귀무가설을 기각시키기 위해 모수 변동시켜보기 
result2 = stats.ttest_1samp(one_sample, popmean=165)
print(result2) # t =3.184, pvalue=0.033, df=4
print(''' 
    유의수준0.05 > pvalue0.033 이므로 귀무가설 기각, 대립가설 채택 
    이 데이터는 우연히 발견된 데이터이다. 따라서 해당 집단의 평균 키가 177아니다.라는 가설 채택
    \n''')

# 시각화하기
# kde가우스 밀도:데이터의 분포를 부드러운 곡선으로 추정해서 그린 것(연속적인 밀도 곡선)
sns.displot(one_sample, bins=10, kde=True) 
plt.xlabel('data')
plt.ylabel('value')
plt.show()