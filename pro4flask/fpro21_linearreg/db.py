'''db에 관련된 별도의 파일 생성'''
import os
import pymysql

# db 연결
def get_conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST","127.0.0.1"),
        port=os.getenv("PORT", 3306),
        user=os.getenv("DB_USER","root"),
        password=os.getenv("DB_PASSWORD",'123'),
        database=os.getenv("DB_NAME","test"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )



# select1 함수 생성
def fetchall_ibsail():
    sql = '''
        select jikwonjik, TRUNCATE((DATEDIFF(NOW(),jikwonibsail))/365,0) AS years,
        jikwonpay as pay from jikwon 
        '''
    
    conn = get_conn()# 연결객체 생성
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()

# select2 함수 생성
def fetchall_avgpay():
    sql = '''
        SELECT jikwonjik  AS 직급 , AVG(jikwonpay) AS 평균연봉 
        FROM jikwon GROUP BY jikwonjik
        '''
    
    conn = get_conn()# 연결객체 생성
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()