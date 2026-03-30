'''
통계적 가설 검정 :
1)
귀무 : 분산은 7이다
대립 : 분산은 7 미만이다

2)
귀무 : 불량률은 3% 이다
대립 : 불량률은 3% 초과이다

3)
귀무 : 기계1에서 생산된 제품의 분산이 크다
대립 : 기계1에서 생산된 제품의 분산이 작거나 같다
------------------------------------------------------------------
카이제곱 검정
카이제곱 문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
예제파일 : cleanDescriptive.csv
칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
조건 :  level, pass에 대해 NA가 있는 행은 제외한다.

카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 
그렇다면 jikwon_jik과 jikwon_pay 간의 관련성 여부를 통계적으로 가설검정하시오.
예제파일 : MariaDB의 jikwon table 
jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
조건 : NA가 있는 행은 제외한다.
'''
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np
import seaborn as sns
import pymysql
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

'''
카이제곱 문제1) 부모학력 수준(x)이 자녀의 진학여부(y)와 관련이 있는가?를 가설검정하시오
예제파일 : cleanDescriptive.csv
칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
조건 :  level, pass에 대해 NA가 있는 행은 제외한다.
level = [1:고졸, 2:대졸, 3:대학원졸]
pass = [1:성공, 2:실패]

귀무가설 : 부모학력 수준이 자녀의 진학여부와 관련이 있다.
대립가설 : 부모학력 수준이 자녀의 진학여부와 관련이 없다.

'''
print('-'*20,'카이제곱 문제1','-'*20)
df1 = pd.read_csv("cleanDescriptive.csv")
print(df1.head(2), df1.columns, type(df1))

# level, pass에 대해 NA가 있는 행은 제외
df1 = df1.dropna(subset=['level','pass'])
# print(df1.head(3), df1['level'].unique(),df1['pass'].unique()) 
# [1. 2. 3.] [2. 1.]

# 부모학력 수준(x)이 자녀의 진학여부(y)에 대한 교차테이블 생성
df1_ctab= pd.crosstab(index=df1['level'], columns=df1['pass'])
df1_ctab.index=['고졸', '대졸', '대학원졸']
df1_ctab.columns=['성공','실패']
print(df1_ctab)

# 이원카이제곱검정
chi2, p, df, exp = stats.chi2_contingency(df1_ctab)
print(f'p={p}')         # p = 0.25070568406521365
print(f'chi2={chi2}')   # chi2 = 2.7669512025956684
print(f'dof={df}')     # dof = 2
print(f'예측된 기대도수(exp)\n{exp}')

print('''
        판정1 : 유의수준0.05 > p-value 0.2507 이므로 귀무가설이 채택됨.
        부모학력 수준이 자녀의 진학여부와 관련이 있다.라는 의견 유지
        판정2 : 임계값=5.99, chi=2.766이므로 통계값은 채택역에 존재하므로 귀무가설 채택
    ''')

# ========================================================================
'''
카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 
그렇다면 jikwon_jik과 jikwon_pay 간의 관련성 여부를 통계적으로 가설검정하시오.
예제파일 : MariaDB의 jikwon table 
jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
조건 : NA가 있는 행은 제외한다.

귀무가설 : A회사의 직급과 연봉은 관련이 없다
대립가설 : A회사의 직급과 연봉은 관련이 있다
'''
print('-'*20,'카이제곱 문제2','-'*20)

# Mariadb jikwon table 읽어오기
db_config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8mb4'
}
def get_connection():
    return pymysql.connect(**db_config)
sql ='''
    select jikwonjik, jikwonpay from jikwon
    '''
try:
    # SQL 실행
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            # 컬럼(c[0])~등에 대한 정보 얻기 : description
            cols = [c[0] for c in cur.description]
            df2 = pd.DataFrame(rows, columns=cols)

            df2.to_csv("jikwon_chi.csv", index=False)
            
except Exception as err :
    print("err : ", err)


df2 = pd.read_csv('jikwon_chi.csv')    
df2 = df2.dropna(subset=['jikwonjik','jikwonpay'])
print(df2.head(2), df2.columns)
print(df2[df2['jikwonjik']=='이사'])

# 데이터 가공하기 - df if문 사용 안하고 하는법
'''
df2.loc[조건, '직급'] = 값 형태는
조건을 만족하는 행만 골라서'직급' 컬럼에 값을 넣는 방식
'''
df2.loc[df2['jikwonjik']=='이사', '직급']=1
df2.loc[df2['jikwonjik']=='부장', '직급']=2
df2.loc[df2['jikwonjik']=='과장', '직급']=3
df2.loc[df2['jikwonjik']=='대리', '직급']=4
df2.loc[df2['jikwonjik']=='사원', '직급']=5
print(df2.head(2))

# 구간나누기(범주화)1. cut()
# print(df2.loc[(df2['jikwonpay'] > 1000) & (df2['jikwonpay'] < 2999)])
bins = [1000, 3000, 5000, 7000, 10000]
lables = [1, 2, 3, 4]
df2['연봉'] = pd.cut(df2['jikwonpay'], bins=bins, labels=lables)

# 구간나누기(범주화)2. 조건
# df2.loc[(df2['jikwonpay'] >= 7000),'연봉']=4
# df2.loc[(df2['jikwonpay'] >= 5000) & (df2['jikwonpay'] < 6999),'연봉']=3
# df2.loc[(df2['jikwonpay'] >= 3000) & (df2['jikwonpay'] < 4999),'연봉']=2
# df2.loc[(df2['jikwonpay'] >= 1000) & (df2['jikwonpay'] < 2999),'연봉']=1

# 구간나누기(범주화)3. apply()
'''
def change_grade(jik):
    if jik == '이사':
        return 1
    elif jik == '부장':
        return 2
    elif jik == '과장':
        return 3
    elif jik == '대리':
        return 4
    else:
        return 5

df2['직급'] = df2['jikwonjik'].apply(change_grade)
'''
print(df2.head(2))

# 검증하기 직급(x), 연봉(y)
df2_ctab = pd.crosstab(index=df2['직급'], columns=df2['연봉'])
df2_ctab.index=['이사', '부장', '과장', '대리', '사원']
df2_ctab.columns=['1000 ~2999', '3000 ~4999', '5000 ~6999', '7000 ~']
print(df2_ctab)

# 이원카이제곱검정
chi2, p, df, exp = stats.chi2_contingency(df2_ctab)
print(f'p={p}')         # p = 0.00029263428943485575
print(f'chi2={chi2}')   # chi2 = 36.27472527472528
print(f'dof={df}')     # dof = 12 
print(f'예측된 기대도수(exp)\n{exp}')

print('''
판정1 : 유의수준0.05 > p-value 0.00029 이므로 귀무가설이 기각됨.
A회사의 직급과 연봉은 관련이 있다라는 대립가설 채택

판정2 : 임계값=21.03, chi=36.27이므로 통계값은 기각역에 존재하므로 대립가설 채택
''')