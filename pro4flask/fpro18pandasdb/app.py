from flask import Flask, render_template, request
import pymysql
import pandas as pd
import numpy as np
from markupsafe import escape # 웹 공격(XSS) 방지 모듈

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
    return render_template('index.html')

@app.route('/dbshow', methods=['GET','POST']) #route 메소드 ['GET','POST']써도 되고 안써도 되고
def dbshow():
    # 부서 검색용 변수 받기
    dept = request.args.get("dept","").strip()

    sql = """
        select j.jikwonno as 직원번호, j.jikwonname as 직원명, b.busername as 부서명,
        b.busertel as 부서전화, j.jikwonpay as 연봉, j.jikwonjik as 직급
        from jikwon j inner join buser b on j.busernum=b.buserno
        """
    
    params=[]
    # dept있을 때 내용을 따라 붙일것이다. 라는 if문
    if dept:
        sql += " where b.busername like %s"
        # params 에 담고
        params.append(f'%{dept}%')

    # sql문 정렬 
    sql += " order by j.jikwonno asc"

    # SQL 실행
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            # 컬럼(c[0])~등에 대한 정보 얻기 : description
            cols = [c[0] for c in cur.description]

    # read_sql은 조건이 많은경우 사용하기 어렵다.
    # 데이터를 처리하기 위해서는 지금 방식이 좋음.
    df = pd.DataFrame(rows, columns=cols)
    
    # 잘 나오나 확인하기 -ok
    # print(df.head())

    """--------- 직원 정보 html로 전송 --------"""
    # 1.df에 정보가 있다면 html로 넘기기위한 준비
    if not df.empty:
        jikwondata = df[['직원번호','직원명','부서명','부서전화','연봉','직급']].to_html(index=False)
    else:
        jikwondata='직원 자료 검색 결과가 없습니다.'
    # print(jikwondata)

    # 2.dbshow.html로 보내야하니까 jikwondata 달고 가자

    """--------- 직급별 연봉 통계(평균 / 표준편차 / 인원수) 넘기기 --------"""
    if not df.empty:
        # groupby를 이용한 DF 생성
        stats_df = (
            df.groupby("직급")['연봉']
            .agg(
                평균 = 'mean',
                # 표준편차는 하나하나 계산해야하니까
                표준편차=lambda x:x.std(ddof=0), # ddof=0 자유도
                인원수='count'
            )
            .round(2)
            .reset_index() # 인덱스로 사용된 칼럼에대해 원래대로 되돌리는방법(...메소드체인 사용중)
            .sort_values(by='평균', ascending=False)
        )
        # 표준편차가 na 일 경우 0으로 채우겠다
        stats_df['표준편차'] = stats_df['표준편차'].fillna(0)
        # print(stats_df)
        statsdata = stats_df.to_html(index=False)
        # print(statsdata)
    else:
        statsdata='통계 대상 자료가 없습니다.'
    return render_template('dbshow.html', 
                            dept=escape(dept), # dept 입력값이 넘나들때 이스케이프 처리를 자동으로 해줌. xss방지
                            jikwondata=jikwondata, # 부서별 직원데이터
                            statsdata=statsdata) # 직급별 연봉 통계 데이터





if __name__ == '__main__':
    app.run(debug=True)