"""
pandas 문제 7)
"""
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
'''
26.03현재 pandas read_sql 사용시 권장하는 방법
pip install sqlalchemy
from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://root:123@127.0.0.1:3306/test")
'''
from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://root:123@127.0.0.1:3306/test")

config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8'
}

try:
    conn = pymysql.connect(**config)
    cursor = conn.cursor()
    '''====================== a ===================='''
    print('-'*15,'사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성','-'*15)
    sql = """
        select j.jikwonno, j.jikwonname, b.busername, j.jikwonpay, j.jikwonjik, j.jikwongen
        from jikwon j inner join buser b on j.busernum=b.buserno
        """
    cursor.execute(sql)
    # execute확인
    # for(jikwonno,jikwonname,busername,jikwonpay,jikwonjik) in cursor:
    #     print(jikwonno,jikwonname,busername,jikwonpay,jikwonjik)
    jikdf = pd.DataFrame(cursor.fetchall(), columns=['사번' ,'이름', '부서명', '연봉', '직급', '성별'])
    print(jikdf.head(3))
    print()
    
    # DataFrame의 자료를 파일로 저장
    print('-'*15,'DataFrame의 자료를 파일로 저장','-'*15)
    jikdf.to_csv('pandasdb_test.csv', index=None, header=None)

    print('-'*15,'부서명별 연봉의 합, 연봉의 최대/최소값을 출력','-'*15)
    # 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
    jikdf2 = pd.read_csv('pandasdb_test.csv',header=None, names=['사번' ,'이름', '부서명', '연봉', '직급','성별'])
    jik_pvt=pd.pivot_table(jikdf2, index='부서명',values='연봉',
                        aggfunc=['sum','max','min'])
    print(jik_pvt)
    print()

    print('-'*15,'부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))','-'*15)
    #부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
    jikcrs= pd.crosstab(jikdf2['부서명'], jikdf2['직급'])
    print(jikcrs)

    print('-'*15,'직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력','-'*15)
    # 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 
    # 담당 고객이 없으면 "담당 고객  X"으로 표시
    sql2 = "select jikwonname, gogekno, gogekname, gogektel from gogek right outer join jikwon on gogek.gogekdamsano=jikwon.jikwonno"
    jik_go = (pd.read_sql(sql2, engine))
    jik_go.gogekno = jik_go.gogekno.fillna('X')
    print(jik_go)
    
    print('-'*15,'연봉 상위 20% 직원 출력','-'*15)
    #- 연봉 상위 20% 직원 출력  : quantile()
    print(jikdf2['이름'][jikdf2['연봉']>=jikdf2['연봉'].quantile(0.8)])
    print()
    
    print('-'*15,'직급별 평균 연봉 출력','-'*15)
    #- SQL로 1차 필터링 후 pandas로 분석 
    # - 조건: 연봉 상위 50% (df['연봉'].median() ) 만 가져오기  
    # 후 직급별 평균 연봉 출력
    jikp = jikdf2[jikdf2['연봉']>=jikdf2['연봉'].quantile(0.5)]
    jik_pvt2=pd.pivot_table(jikp, index='직급',values='연봉',
                        aggfunc=['mean']).reset_index()
    
    dept_pvt = pd.pivot_table(data=jikp, index='부서명', values='연봉').reset_index()
    print(dept_pvt)
    

    print('-'*15,'부서명별 연봉의 평균으로 가로 막대 그래프를 작성','-'*15)
    #부서명별 연봉의 평균으로 가로 막대 그래프를 작성
    plt.barh(dept_pvt['부서명'], dept_pvt['연봉'])
    plt.show()

    '''====================== b ===================='''
    print('-'*15,'b_pivot_table을 사용하여 성별 연봉의 평균을 출력','-'*15)
    jik_pvt3 = pd.pivot_table(
        data=jikdf, index='성별', values='연봉' ).reset_index()
    print(jik_pvt3)
    print()

    print('-'*15,'b_성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프','-'*15)
    plt.bar(jik_pvt3['성별'],jik_pvt3['연봉'])
    plt.show()
    print('-'*15,'b_부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))','-'*15)
    jikcis2 = pd.crosstab(jikdf['부서명'], jikdf['성별'])
    print(jikcis2)
    print()

    '''====================== c ===================='''
    print('-'*15,'c_키보드로 사번, 직원명을 입력받아 로그인에 성공하면 console에 아래와 같이 출력','-'*15)
    no = input('사번')
    name = input('직원명')
    sql3 = f'''select jikwonno, jikwonname, busername, jikwonjik, busertel, jikwongen 
    from jikwon inner join buser on jikwon.busernum=buser.buserno 
    where jikwon.jikwonno={no} and jikwon.jikwonname='{name}' 
    '''
    inputdf = pd.read_sql(sql3, engine)
    print(inputdf,'\n인원수 : ',len(inputdf))
    
    print('-'*15,'c_성별 연봉 분포 + 이상치 확인','-'*15)
    gen1 = jikdf[jikdf['성별']=='여']
    gen2 = jikdf[jikdf['성별']=='남']
    figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    sns.boxplot(data=jikdf, x='성별', y='연봉', ax=ax1) #연도별 대여 횟수
    sns.barplot(data=jikdf, x='성별', y='연봉', ax=ax2)
    # plt.scatter(jikdf['성별'], jikdf['연봉'])
    plt.show()

    # print(gen1, gen2)
    print('-'*15,'c_Histogram (분포 비교) : 남/여 연봉 분포 비교','-'*15)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.hist(gen1['연봉'], bins=15)
    ax2.hist(gen2['연봉'], bins=15)
    ax1.set(ylabel=' 인원수', title='여성 연봉 분포(histogram)')
    ax2.set(ylabel=' 인원수', title='남성 연봉 분포(histogram)')
    plt.show()


except Exception as err:
    print("err : ", err)
finally:
    cursor.close()
    conn.close()
