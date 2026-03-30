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
        database=os.getenv("DB_NAME","coffeedb"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )

# 웹에서 가져온 설문조사 결과를 insert 함수 생성
def insert_survey( gender:str,age:int,co_survey:str)-> None: # 반환값은 없어
    # gender:str | age:int  | co_survey:str
    sql = "insert into survey(gender, age, co_survey) values (%s, %s, %s)"
    conn = get_conn() # 연결객체 생성
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (gender, age, co_survey))
    finally:
        conn.close()


# select 함수 생성
def fetchall_survey() -> list[dict]: # -> 반환값 type 힌트 적기
    sql = 'select rnum,gender,age,co_survey from survey order by rnum asc'
    
    conn = get_conn()# 연결객체 생성
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()