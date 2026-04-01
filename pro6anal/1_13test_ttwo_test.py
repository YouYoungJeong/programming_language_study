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

귀무가설 : 포장지 색상에 따른 제품의 매출액에 차이가 존재 하지 않다
대립가설 : 포장지 색상에 따른 제품의 매출액에 차이가 존재 한다.

수집된 자료 :  
    blue : 70 68 82 78 72 68 67 68 88 60 80
    red : 60 65 55 58 67 59 61 68 77 66 66
'''
print('='*30,'[two-sample t 검정 : 문제1] ','='*30)
blue = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
red = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]

# 데이터 전처리
print('-'*20,' 데이터 전처리 ','-'*20)
print("blue 의 평균 : ",round(np.mean(blue), 2))
print("red 의 평균 : ",round(np.mean(red), 2))
print("blue와 red포장지의 평균 차이 : ",round(np.mean(blue)-np.mean(red), 2))
print("수치상 평균의 차이가 유의한지 안한지 알아봐야함.")
print()

# 정규성 검정
print('-'*20,' 정규성 검정 ','-'*20)
print("blue 정규성 검사(Shapiro) :",stats.shapiro(blue).pvalue)
print("red 정규성 검사(Shapiro) :",stats.shapiro(red).pvalue)
print("두그룹 정규성을 만족")
print()

# 등분산성 검정
print('-'*20,' 등분산성 검정 ','-'*20)
print(levene(blue, red).pvalue) # 0.4391644468508382
print('두 그룹 등분산성을 만족\n')

# t검정
print('-'*20,'independent two samples t-test','-'*20)
print(stats.ttest_ind(blue, red, equal_var=True))
# statistic=2.9280203, pvalue=0.0083165, df=20
print()

# 결과
print('-'*20,'결과','-'*20)
print('''
    판단1 - pvalue)
        유의확률(α)0.05 > pvalue=0.0083165 이므로 귀무가설을 기각한다.
        따라서 포장지 색상에 따른 제품의 매출액에 차이가 존재 한다.라는 주장을 받아들임
    판단2 - 임계값)
        임계치:1.725 < t통계량=2.9280203 t통계량이 기각역에 존재하여 귀무가설을 기각.
    ''')
print()

'''
[two-sample t 검정 : 문제2] ================================================================== 
아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 
혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
수집된 자료 :  
    남자 : 0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8
    여자 : 1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4

귀무 가설 : 성별과 혈관 내의 콜레스테롤 양에 차이가 없다.
대립 가설 : 성별과 혈관 내의 콜레스테롤 양에 차이가 있다.
'''
print('='*30,'[two-sample t 검정 : 문제2] ','='*30)
print('-'*20,' 데이터 전처리 ','-'*20)
male =  [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
female =[1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]

male = np.random.choice(male, size=15, replace=True)
female = np.random.choice(female, size=15, replace=True)
# 평균값 확인하기
print(f"남성 평균 : {np.mean(male)}, 여성 평균 :{np.mean(female)},\
        \n남성과 여성의 평균 차이 : {np.mean(male)-np.mean(female)}\n" )

# 정규성 검정
print('-'*20,' 정규성 검정 ','-'*20)
print("male-shapiro-pvalue : ",stats.shapiro(male).pvalue)  #  0.01670927366885352
print("female-shapiro-pvalue : ",stats.shapiro(female).pvalue) # 0.000801368079200278

# 정규성 검정 시각화하기
# Q-Q plot확인하기
# stats.probplot(male, plot=plt)
# plt.title("male Q-Q plot")
# plt.show()
# stats.probplot(female, plot=plt)
# plt.title("female Q-Q plot")
# plt.show()

print("shapiro검정 p-value와 Q-Q plot그래프를 확인한 결과\n\
    male 데이터는 정규성을 따르지만 female 데이터는 정규성을 따르지 않는다.\n" \
    "하지만 둘중 하나라도 정규성을 따르기 때문에 t-검정 사용이 가능하다고 봄\n")
print()

# 등분산성 검정
print('-'*20,' 등분산성 검정 ','-'*20)
print(levene(male, female)) # pvalue= 0.48643817002986545
print("두데이터 모두 등분산성을 따른다.")
print()

#t-test
print('-'*20,'independent two samples t-test','-'*20)
print(stats.ttest_ind(male, female, equal_var=True))
# statistic=-1.697457002, pvalue=0.10070002873, df=28
print()

# Mann–Whitney U test
print('-'*20,'Mann-Whitney U test','-'*20)
print(stats.mannwhitneyu(male, female))
# statistic=96, pvalue=0.5061570699
print()

# 결과
print('-'*20,'결과','-'*20)
print('''
    판단1 - pvalue)
        유의확률(α)0.05 < pvalue=0.1007 이므로 귀무가설을 유지한다.
        Mann–Whitney U test 검증 시행결과 역시 유의확률(α)0.05 < pvalue=0.5061570이므로 귀무채택
        따라서 성별과 혈관 내의 콜레스테롤 양에 차이가 없다.라는 의견 유지
    판단2 - 임계값)
        임계치:1.701 > t통계량=-1.69745 이므로 t값이 채택력안에 존재하므로 귀무채택
    ''')
print()

'''
[two-sample t 검정 : 문제3]===============================================================
DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 
연봉의 평균에 차이가 존재하는지 검정하시오.
연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.

귀무가설 : 직급에 따른 연봉의 평균차이가 존재하지 않는다 
대립가설 : 직급에 따른 연봉의 평균차이가 존재한다
'''
print('='*30,'[two-sample t 검정 : 문제3] ','='*30)

# 데이터 전처리
print('-'*20,' 데이터 전처리 ','-'*20)
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
    on busernum=buserno where buserno=10 or buserno=20;
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
            df.to_csv("jikwon_t.csv", index=False)
            
except Exception as err :
    print("err : ", err)

# 데이터 정제 시작
data = pd.read_csv("jikwon_t.csv")
jik_y = data[data['busername']=='영업부']
jik_c = data[data['busername']=='총무부']

# 데이터 확인
print(jik_y['busername'].unique(),len(jik_y['busername']), jik_y.isnull().sum())
print(jik_c['busername'].unique(), len(jik_c['busername']), jik_c.isnull().sum())
print()

# 평균값 확인하기
print(f'총무부의 연봉 평균 : {jik_c['jikwonpay'].mean()}')
print(f'영엽부의 연봉 평균 : {jik_y['jikwonpay'].mean()}')
print(f'총무부-영업부 연봉 차이 : {jik_c['jikwonpay'].mean()-jik_y['jikwonpay'].mean()}')
print()

# 정규성 검정하기
print('-'*20,' 정규성 검정 ','-'*20)
print("총무부의 연봉 shapiro-pvalue : ",stats.shapiro(jik_c['jikwonpay']).pvalue) # 0.026044936412817302
print("영업부의 연봉 shapiro-pvalue : ",stats.shapiro(jik_y['jikwonpay']).pvalue) # 0.025608399511523605

# 정규성 시각화 확인1. (hist) 
# sns.histplot(np.array(jik_c['jikwonpay']), bins=10, kde=True)
# sns.histplot(np.array(jik_y['jikwonpay']), bins=10, kde=True)
# plt.show()

# 정규성 시각화 확인2. (Q-Q plot)
# stats.probplot(np.array(jik_c['jikwonpay']), plot=plt)
# plt.title("총무부 Q-Q plot")
# plt.show()
# stats.probplot(np.array(jik_y['jikwonpay']), plot=plt)
# plt.title("영업부 Q-Q plot")
# plt.show()

print("두 그룹 모두 shapiro의 pvalue값은 정규성을 따르지 않는다.\n" \
    "정규성 데이터의 시각화는 데이터역시 정규성을 따르지 않음.")
print()

# 등분산성 검정
print('-'*20,' 등분산성 검정 ','-'*20)
print(levene(jik_c['jikwonpay'], jik_y['jikwonpay']).pvalue) # 0.915044305043978
print("두 그룹 모두 levene의pvalue값이 유의확률(α)0.05을 초과하기 때문에 등분산성을 만족함.")
print()

# Mann–Whitney U test
print('-'*20,'Mann-Whitney U test','-'*20)
print(stats.mannwhitneyu(jik_c['jikwonpay'], jik_y['jikwonpay']))
# statistic=51.0, pvalue=0.472133
print()
# 결과
print('-'*20,'결과','-'*20)
print('''
    판단1 - pvalue)
        유의확률(α)0.05 < pvalue=0.472133 이므로 귀무가설을 유지한다.
        따라서 직급에 따른 연봉의 평균차이가 존재하지 않는다.라는 의견 유지
    ''')
print()

'''
[대응표본 t 검정 : 문제4]====================================================================
어느 학급의 교사는 매년 학기 내 치뤄지는 시험성적의 결과가 실력의 차이없이 
비슷하게 유지되고 있다고 말하고 있다. 
이 때, 올해의 해당 학급의 중간고사 성적과 기말고사 성적은 다음과 같다. 
점수는 학생 번호 순으로 배열되어 있다.
수집된 자료 :  
    중간 : 80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80
    기말 : 90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95
그렇다면 이 학급의 학업능력이 변화했다고 이야기 할 수 있는가?

귀무가설 : 매년 학기 치뤄지는 시험성적의 결과의 변화가 없다
대립가설 : 매년 학기 치뤄지는 시험성적의 결과의 변화가 있다
'''
print('='*30,'[대응표본 t 검정 : 문제4] ','='*30)

# 데이터 확인
print('-'*20,' 데이터 확인 ','-'*20)
mid = [80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80]
final = [90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95]
print("중간고사 평균 :",np.mean(mid), "기말고사 평균 : " ,np.mean(final), 
        "\n 중간고사 평균-기말고사 평균",np.mean(mid)-np.mean(final))
print()

# 정규성 검정
print('-'*20,' 정규성 검정 ','-'*20)
print("중간고사-shapiro-pvalue : ",stats.shapiro(mid).pvalue)    # 0.3681471063353156
print("기말고사-shapiro-pvalue : ",stats.shapiro(final).pvalue)   # 0.1930029726717273

# 정규성 시각화 확인1. (hist) 
# sns.histplot(mid, bins=10, kde=True)
# sns.histplot(final, bins=10, kde=True)
# plt.title("중간, 기말 hist")
# plt.show()

# 정규성 시각화 확인2. (Q-Q plot)
# stats.probplot(mid, plot=plt)
# plt.title("중간고사 Q-Q plot")
# plt.show()
# stats.probplot(final, plot=plt)
# plt.title("기말고사 Q-Q plot")
# plt.show()

print('히스토그램에 완만하지만 정규분포그래프의 모양을 따르고 있고' \
        'shapiro검정 pvalue값에의해 정규성을 만족함.')

# paired samples t-test
print('-'*20,'paired samples t-test','-'*20)
print(stats.ttest_rel(mid, final))
# statistic=-2.628112, pvalue=0.02348619, df=11

# 결과
print('-'*20,'결과','-'*20)
print('''
    판단1 - pvalue)
        유의확률(α)0.05 > pvalue=0.02348619 이므로 귀무가설을 기각한다
        따라서 매년 학기 치뤄지는 시험성적의 결과의 변화가 있다고 판단하여
        이 학급의 학업능력이 변화했다고 이야기 할 수 있다. 
    판단2 - 임계값)
        임계치:1.796 > t통계량=-2.628112 이므로 t값이 기각력안에 존재하므로 대립가설을 채택한다.
    ''')