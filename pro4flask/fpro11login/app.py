from flask import Flask, render_template, request, redirect, url_for, flash, session
import pymysql
import os
from flask import get_flashed_messages
"""
    env에서 secret_key, MariaDB 연결 정보 다 읽어올 수 있다
    env 파일 읽어오는 모듈 설치
        pip install python-dotenv
"""
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = "abcdef123456" # session/flash를 위한 쿠키 서명용 secret_key

load_dotenv() # .env파일에 저장된 환경변수 읽는 함수

# MariaDB 연결 정보
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

def get_conn():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4", # utf8-한글처리 utf8mb4-전세계문자+이모지 까지 처리
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )

@app.get("/")
def root():
    return redirect(url_for("login_form"))

""" 
    login form 
    get. post둘다 들어옴.
"""
@app.get("/login")
def login_form():
    return render_template("login.html")


# 직원정보 받기
@app.post("/login")
def login_post():
    jikwonno_raw = (request.form.get("jikwonno") or "").strip()
    jikwonname = (request.form.get("jikwonname") or "").strip()
    
    if not jikwonno_raw.isdigit() or not jikwonname:
        flash("직원번호는 숫자, 직원이름은 필수 입니다.")
        return redirect(url_for("login_form"))
    
    jikwonno = int(jikwonno_raw)

    conn = get_conn()
    try:
        with conn.cursor() as cur:      
            # 로그인 체크
            cur.execute("""
                select jikwonno, jikwonname from jikwon where jikwonno=%s and jikwonname=%s
            """,(jikwonno, jikwonname))
            
            me = cur.fetchone()
            if not me:
                flash("로그인 실패 : 직원정보 불일치.")
                return redirect(url_for("login_form"))
            
            # 로그인 성공인 경우
            cur.execute("""
                select jikwonno, jikwonname, jikwonjik, busername, jikwonpay,
                year(jikwonibsail) as jinwonibsil_year
                from jikwon inner join buser on busernum=buserno
                order by jikwonno
                """)
            rows = cur.fetchall()

            # 세션 생성
            session["jikwonno"] = me["jikwonno"]
            session["jikwonname"] = me["jikwonname"]
            print(me) # {'jikwonno': 1, 'jikwonname': '홍길동'}
            return render_template("jikwonlist.html", rows=rows, login_user=me)
            
    except Exception as e:
        print("에러 발생:", e)
        flash(f"DB 오류: {e}")
        return redirect(url_for("login_form"))
    finally:
        conn.close()

# 세션체크(login)후 담당고객 출력하기
@app.get("/gogek/<int:jikwonno>")
def gogek_list(jikwonno:int):
    if "jikwonno" not in session:
        flash("로그인 후 고객정보 이용하세요")
        return redirect(url_for("login_form"))
    
    conn= get_conn()
    try:
        with conn.cursor() as cur:      
            # 고객정보 확인
            cur.execute("""
                select gogekno, gogekname, gogektel
                from gogek where gogekdamsano=%s order by gogekno
            """,(jikwonno,)) #tuple!! 
            customers = cur.fetchall()  
            # 직원 정보 확인
            cur.execute("""
                select jikwonname from jikwon where jikwonno=%s
                """,(jikwonno,))
            emp = cur.fetchone()
            
        return render_template("gogkelist.html", 
                                customers=customers, 
                                empno=jikwonno, 
                                empname=(emp['jikwonname'] if emp else ""))
    finally:
        conn.close()
    
""" 고객리스트에서 직원리스트 보기 """
@app.get("/jikwons")
def jikwon_list():
    if "jikwonno" not in session:
        flash("로그인 후 이용하세요")
        return redirect(url_for("login_form"))
    conn= get_conn()

    try:
        with conn.cursor() as cur:      
            cur.execute("""
                select jikwonno, jikwonname, jikwonjik, busername, jikwonpay,
                year(jikwonibsail) as jinwonibsil_year
                from jikwon inner join buser on busernum=buserno
                order by jikwonno
                """)
            rows = cur.fetchall()
        # login이 끝난상태에서 login정보 세션값 받아 jikwons page로 넘어가기
        login_user = {"jikwonno":session["jikwonno"],"jikwonname":session["jikwonname"]}
        return render_template("jikwonlist.html", rows=rows, login_user=login_user)

    finally:
        conn.close()

""" logout 만들기 """
@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_form"))


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")