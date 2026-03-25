'''
MariaDB에 저장된 jikwon, buser 테이블을 이용하여 아래의 문제에 답하시오.
'''

from flask import Flask, render_template, request
import pymysql
import pandas as pd
import numpy as np
from markupsafe import escape # 웹 공격(XSS) 방지 모듈
import matplotlib.pyplot as plt
import koreanize_matplotlib

app = Flask(__name__)

# 비권장 .env사용하기
db_config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8mb4'
}
# DB 연결 함수 생성
def get_connection():
    return pymysql.connect(**db_config)

#templates 폴더명 제대로 안주면 err!
@app.route('/')
def index():
    return render_template('main.html')

@app.route('/showdb')
def showdb():
    # 부서 검색용 변수 받기
    dept = request.args.get("dept","").strip()
    # print(dept) # 들어오는거 확인
    '''1) 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. (join)
    : 부서번호, 직원명 순으로 오름 차순 정렬 '''
    sql = """
        select jikwonno as 사번, jikwonname as 직원명, busername as 부서명, 
        jikwonjik as 직급, jikwonpay as 연봉, 
        TIMESTAMPDIFF(YEAR, jikwonibsail, NOW()) as 근무년수 
        from jikwon inner join buser on jikwon.busernum=buser.buserno
        order by buser.buserno, jikwon.jikwonname 
        """

    # SQL 실행
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            # 컬럼(c[0])~등에 대한 정보 얻기 : description
            cols = [c[0] for c in cur.description]

    # read_sql은 조건이 많은경우 사용하기 어렵다.
    # 데이터를 처리하기 위해서는 지금 방식이 좋음.
    df = pd.DataFrame(rows, columns=cols)
    # print(df)
    """--------- 직원 정보 html로 전송 --------"""
    # 1.df에 정보가 있다면 html로 넘기기위한 준비
    if not df.empty:
        jikwondata = df[['사번', '직원명', '부서명', '직급', '연봉', '근무년수']].to_html(index=False)
    else:
        jikwondata='직원 자료 검색 결과가 없습니다.'
    # print(jikwondata)

    '''2) 부서명, 직급 자료를 이용하여  각각 연봉합, 연봉평균을 구하시오.'''
    # 2-1)
    buser_des = (df.groupby('부서명')['연봉'].agg(
        연봉합 = 'sum',
        연봉평균='mean'
    ).reset_index().round(2))
    buser_des_html = buser_des.to_html(index=False)
    # print(buser_des)
    # 2-2)
    jik_des = (df.groupby('직급')['연봉'].agg(
        연봉합 = 'sum',
        연봉평균='mean'
    ).reset_index().round(2)).to_html(index=False)
    # print(jik_des)

    '''3) 부서명별 연봉합, 평균을 이용하여 세로막대 그래프를 출력하시오.'''
    """fig = plt.figure()
    ax1 = plt.bar(buser_des['부서명'],buser_des['연봉합'] )
    ax2 = plt.bar(buser_des['부서명'],buser_des['연봉평균'] )
    # plt.show()"""
    x = np.arange(len(buser_des['부서명']))
    width = 0.35

    fig = plt.figure(figsize=(5, 3))
    plt.bar(x - width/2, buser_des['연봉합'], width=width, label='연봉합')
    plt.bar(x + width/2, buser_des['연봉평균'], width=width, label='연봉평균')

    plt.xticks(x, buser_des['부서명'])
    plt.xlabel('부서명')
    plt.ylabel('금액')
    plt.title('부서별 연봉합과 연봉평균')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('C:/work/projects/pro4flask/fpro18pandasdb_test/static/img/plot2.png')
    
    '''4) 성별, 직급별 빈도표를 출력하시오'''
    sql2 = '''
            select jikwongen as 성별, jikwonjik as 직급 from jikwon 
        '''
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql2)
            rows2 = cur.fetchall()
            # 컬럼(c[0])~등에 대한 정보 얻기 : description
            cols2 = [c[0] for c in cur.description]
    df2 = pd.DataFrame(rows2, columns=cols2)
    count_result = pd.crosstab(df2['직급'], df2['성별'])
    print(count_result.reset_index())
    df2_w = df2[df2['성별']=='여']
    df2_m = df2[df2['성별']=='남']
    # print(df2)
    count_1 = df2_w.groupby('직급')['성별'].agg(
        여성빈도 = 'count'
    )
    count_2 = df2_m.groupby('직급')['성별'].agg(
        남성빈도 = 'count'
    )
    count_html = pd.concat([count_2, count_1], axis=1).fillna(0).reset_index().to_html(index=False)
    # print(count_html)
    '''5) 부서별 최고 연봉자 출력 : 부서명별로 가장 연봉이 높은 직원 1명씩 출력 
    출력 항목: 부서명, 직원명, 연봉'''
    df3 = df[['부서명','직원명','연봉']]
    df3max = df3.groupby('부서명')['연봉'].agg(
        연봉='max'
    ).reset_index()
    df3max_html= df3.merge(df3max, how='inner').to_html(index=False)


    '''6) 부서별 직원 비율 계산 : 전체 인원 대비 각 부서 인원 비율(%) 
    비율 계산 (%)은 dept_ratio = (dept_count / total * 100).round(2)
    결과를 DataFrame으로 출력
    예: 총 인원: 30명
        영업부 20%
        총무부 30%
        전산부 5%'''
    jik_count = df['부서명'].value_counts().reset_index()
    jik_total = df['부서명'].count()
    jik_count['count']= (jik_count['count'] / jik_total * 100).round(2)
    jik_count.columns = ['부서명','직원비율']
    jik_count_html = jik_count.to_html(index=False)
    # print(jik_count)

    return render_template('showdb.html',dept=escape(dept),jikwondata=jikwondata,
                            buser_des_html= buser_des_html, jik_des=jik_des,
                            count_html=count_html, df3max_html=df3max_html,
                            jik_count_html=jik_count_html)


if __name__ == '__main__':
    app.run(debug=True)