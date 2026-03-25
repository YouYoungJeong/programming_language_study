"""
원격 DB (mariadb) 연동 후 jikwon자료를 읽어 DataFrame에 자료 저장
"""
import MySQLdb # 둘중 누구를 써도 상관 없어.
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import csv
# config, pickle, .env 등등 정보 전달하는 방법
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
    sql = '''
            select jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay
            from jikwon inner join buser on jikwon.busernum = buser.buserno
        '''
    cursor.execute(sql)

    # 데이터 잘 들어오는지 확인
    # for (jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay) in cursor:
    #     print(jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay)
    
    # DataFrame으로 출력하기
    df1 = pd.DataFrame(cursor.fetchall(), columns=['jikwonno', 'jikwonname', 'busername', 'jikwonjik', 'jikwongen', 'jikwonpay'])
    print(df1.head(3))
    print('연봉의 총합 : ', df1['jikwonpay'].sum())
    print()

    # csv파일로 저장,출력하기 - csv file i/o
    cursor.execute(sql)
    with open('pandasdb2.csv', mode='w', encoding='utf-8') as fobj:
        writer = csv.writer(fobj)
        for row in cursor.fetchall():
            writer.writerow(row)
    df2 = pd.read_csv('pandasdb2.csv', header=None, 
                    names=['번호','이름','부서','직급','성별','연봉'])
    print(df2.head(3))
    print()

    # pandas의 sql처리 함수 이용
    print('-'*15,'pandas의 sql처리 함수 이용 - read_sql','-'*15)
    df = pd.read_sql(sql, conn)
    df.columns = ['번호','이름','부서','직급','성별','연봉']
    print(df.head(5))
    print(df[:5])
    print(df[:-21])
    print()
    # 건수
    print(df['이름'].count(), " ", len(df))
    print('부서별 인원수 : ', df['부서'].value_counts())
    print()
    # 조건검색
    print('연봉 7000 이상 :\n', df.loc[df['연봉'] >= 7000])
    print()

    # pivot_table() 대신 성별 직급별 교차표 - crosstab
    ctab = pd.crosstab(df['성별'], df['직급'], margins=True) # margins = 계
    print('교차표 : \n',ctab)

    # 시각화
    jik_ypay = df.groupby(['직급'])['연봉'].mean() # 직급별 연봉 평균
    print("jik_ypay : ", jik_ypay)
    plt.pie(jik_ypay, explode=(0.2, 0, 0,  0.3, 0),
            labels=jik_ypay.index, shadow=True, counterclock=False,
            colors=['magenta', 'aqua', 'green']) # counterclock=False시계방향
    plt.show()
    



except Exception as err:
    print('처리 오류 : ', err)

finally:
    cursor.close()
    conn.close()