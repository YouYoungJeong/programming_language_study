from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from scipy.stats import levene, bartlett, fligner
import pymysql
'''
[two-sample t 검정 : 문제1] 
다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.
수집된 자료 :  
    blue : 70 68 82 78 72 68 67 68 88 60 80
    red : 60 65 55 58 67 59 61 68 77 66 66
'''
print('='*30,'[two-sample t 검정 : 문제1] ','='*30)
blue = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
red = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]

print('-'*20,' 데이터 전처리 ','-'*20)
print('-'*20,' 정규성 검정 ','-'*20)
print('-'*20,' 등분산성 검정 ','-'*20)
print('-'*20,'independent two samples t-test','-'*20)
'''
[two-sample t 검정 : 문제2]  
아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 
혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
수집된 자료 :  
    남자 : 0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8
    여자 : 1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4
'''
print('='*30,'[two-sample t 검정 : 문제2] ','='*30)
print('-'*20,' 데이터 전처리 ','-'*20)
print('-'*20,' 정규성 검정 ','-'*20)
print('-'*20,' 등분산성 검정 ','-'*20)
print('-'*20,'independent two samples t-test','-'*20)
'''
[two-sample t 검정 : 문제3]
DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 
연봉의 평균에 차이가 존재하는지 검정하시오.
연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.
'''
print('='*30,'[two-sample t 검정 : 문제3] ','='*30)
print('-'*20,' 데이터 전처리 ','-'*20)
print('-'*20,' 정규성 검정 ','-'*20)
print('-'*20,' 등분산성 검정 ','-'*20)
print('-'*20,'independent two samples t-test','-'*20)