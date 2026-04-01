import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pymysql
'''
[ANOVA 예제 1]
빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.

귀무가설 : 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양이 같다
대립가설 : 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양이 다르다
'''
# 데이터 정제 하기
print("데이터 정제 하기")
data = pd.read_table("ANOVAquiz1.txt", sep=" ")
print(type(data),"\n" ,data.describe(), "\n" ,data.columns,"\n",len(data))
print("kind.unique() : ",data.kind.unique())

# 각 컬럼별 결측치 확인
print("kind.isna():",data.kind.isna().sum(),"quantity.isna():", data.quantity.isna().sum())

# 결측치 처리(평균값)
data.fillna({'quantity':data['quantity'].mean()}, inplace=True)
print("kind.isna():",data.kind.isna().sum(),"quantity.isna():", data.quantity.isna().sum())

# 그룹화
group1 = data[data['kind']==1]['quantity']
group2 = data[data['kind']==2]['quantity']
group3 = data[data['kind']==3]['quantity']
group4 = data[data['kind']==4]['quantity']
print(group1.head(2), len(group1))
print(group2.head(2), len(group2))
print(group3.head(2), len(group3))
print(group4.head(2), len(group4))

# 네그룹 시각화
# plt.boxplot([group1, group2, group3, group4])
# plt.show()

# 정규성 확인하기 - 만족O
print('정규성 확인하기')
print(stats.shapiro(group1).pvalue) # 0.8680405840743664
print(stats.shapiro(group2).pvalue) # 0.5923924912154501
print(stats.shapiro(group3).pvalue) # 0.48601083943678747
print(stats.shapiro(group4).pvalue) # 0.4162161718602888
print("네그룹 모두 정규성을 만족함.")
print()

# 등분산성 확인하기 - 만족O
print('등분산성 확인하기')
print("levene-pvalue :", stats.levene(group1, group2, group3, group4).pvalue) # 0.326896993
print("bartlett-pvalue :",stats.bartlett(group1, group2, group3, group4).pvalue) # 0.19342011
print()

print("일원분산분석 방법 1 - anova_lm()")
lm_model = ols('quantity ~ C(kind)', data=data).fit()
print(anova_lm(lm_model, typ=1))
print("해석 : p = 0.848244 > α:0,05 이므로 귀무유지.")
print()

# 검정하기 - f_oneway
print("일원분산분석 방법 2 - f_oneway")
print(stats.f_oneway(group1, group2, group3, group4))
# statistic=0.2669351, pvalue=0.84824366
print("해석 : pvalue=0.84824366 > α:0,05 이므로 귀무가설 유지" \
"\n빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양이 같다.라는 의견 유지\n")

# 사후 검정 하기 - 귀무채택 이므로 평균의 차이가 없다.
print("사후 검정 하기")
tukResult = pairwise_tukeyhsd(endog=data.quantity, groups=data.kind, alpha=0.05)
print(tukResult)
# reject -> 전부 False

# 사후 검정 시각화하기
# tukResult.plot_simultaneous(xlabel="mean", ylabel="group")
# plt.show()

'''
[ANOVA 예제 2]
DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 
연봉의 평균에 차이가 있는지 검정하시오. 
만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.

귀무가설 : 직급에 따른 연봉의 평균차이가 존재하지 않는다 
대립가설 : 직급에 따른 연봉의 평균차이가 존재한다
'''
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
    select busername, jikwonpay from jikwon inner join buser 
    on busernum=buserno
    '''
try:
    # SQL 실행
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            # 컬럼(c[0])~등에 대한 정보 얻기 : description
            cols = [c[0] for c in cur.description]
            df = pd.DataFrame(rows, columns=cols)

            # sql문 csv저장
            df.to_csv("jikwon_a.csv", index=False)
            
except Exception as err :
    print("err : ", err)

# 데이터 정제 시작
data = pd.read_csv("jikwon_a.csv")
print(data.busername.unique(),"\n",data.isnull().sum())

# 결측치 처리하기
data = data.dropna()

# 그룹화 하기
jik_y = data[data['busername']=='영업부']['jikwonpay']
jik_c = data[data['busername']=='총무부']['jikwonpay']
jik_j = data[data['busername']=='전산부']['jikwonpay']
jik_k = data[data['busername']=='관리부']['jikwonpay']

# boxplot확인
# plt.boxplot([jik_y,jik_c,jik_j,jik_k])
# plt.show()

# 정규성 확인하기 - 만족O
print('정규성 확인하기')
print(stats.shapiro(jik_y).pvalue) # 0.025608399511523605
print(stats.shapiro(jik_c).pvalue) # 0.026044936412817302
print(stats.shapiro(jik_j).pvalue) # 0.41940720517769636
print(stats.shapiro(jik_k).pvalue) # 0.9078027897950541
print("영업부, 총무부는 정규성을 만족하지 않으나, 전산부와 관리부는 정규성은 만족 " \
"\n반이상이 정규성을 만족하므로 정규성을 만족한다고 하겠다.")
print()

# 등분산성 확인하기 - 만족O
print('등분산성 확인하기')
print("levene-pvalue :", stats.levene(jik_y, jik_c, jik_j, jik_k).pvalue) # 0.7980753526275928
print("bartlett-pvalue :",stats.bartlett(jik_y, jik_c, jik_j, jik_k).pvalue) # 0.33583085291459547
print("정규성을 만족한다고 가정하여 진행한 등분산성bartlett결과 역시 등분산성을 만족한다고 결과가 나옴")
print()

print("일원분산분석 방법 1 - anova_lm()")
lm_model = ols('jikwonpay ~ C(busername)', data=data).fit()
print(anova_lm(lm_model, typ=1))
print("해석 : p = 0.412441 > α:0,05 이므로 귀무유지.")
print()

# 검정하기 - f_oneway
print("일원분산분석 방법 2 - f_oneway")
print(stats.f_oneway(jik_y, jik_c, jik_j, jik_k))
# statistic=0.41244077, pvalue=0.74544218
print("해석 : pvalue=pvalue=0.74544218 > α:0,05 이므로 귀무가설 유지" \
"\n직급에 따른 연봉의 평균차이가 존재하지 않는다.라는 의견 유지\n")

# 정규성 깨지면 Kruskal-Wallis - stats.kruskal() 사용
print("정규성 판단이 애매해 Kruskal-Wallis - stats.kruskal() 사용")
print(stats.kruskal(jik_y, jik_c, jik_j, jik_k))
#KruskalResult:  statistic=1.671252, pvalue=0.6433438 - 귀무채택
print()

# 사후 검정 하기
print("사후 검정 하기")
tukResult = pairwise_tukeyhsd(endog=data['jikwonpay'], groups=data['busername'], alpha=0.05)
print(tukResult)
# reject -> 전부 False

# 사후 검정 시각화하기
# tukResult.plot_simultaneous(xlabel="mean", ylabel="group")
# plt.show()