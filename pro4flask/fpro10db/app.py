from flask import Flask, render_template, request, redirect, url_for, flash
import pymysql
import os
from flask import get_flashed_messages
"""
- flash : 
    임시메세지(내부적으로 Session에 저장해 둠), 
    내가 만든 세션에 저장하기 때문에 *secret_key*를 줘야함 
    -> 세션을 쓰겠다는게 아니라 임시메세지를 사용하기위해
    session/flash를 위한 쿠키 서명용 secret_key
    ex)flash("err") - 메세지를 세션에 *잠시* 저장

- get_flashed_messages :
    저장해둔 메세지를 꺼내는 함수,
    메세지를 읽어오면 세션에서 없어짐   
    ex)get_flashed_messages() - flash로 저장해둔 메세지를 읽음,
    
    

- pip install pymysql - pymysql 모듈 사용
"""
app = Flask(__name__)
app.secret_key = "abcdef123456" # session/flash를 위한 쿠키 서명용 secret_key

""" MariaDB 연결 정보 
        원래는 감춰야함 - 피클
        env파일로 읽어와도 됨. 
"""
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123")
DB_NAME = os.getenv("DB_NAME", "test")

def get_conn():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4", # utf8-한글처리 utf8mb4-전세계문자+이모지 까지 처리
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )
# DictCursor : select결과를 "dict type" 형태로 받게해주는 작업을 함
#{'code':1 , 'sang':'마우스'..}로 받아서 -> row['code'],row['sang']로 처리



@app.get("/")
def root():
    return redirect(url_for("show_list"))


""" DB연동후 list.html로 보내 상품목록 내용 채우기 """
@app.get('/show/')
def show_list():
    # DB와 연결
    conn = get_conn()
    
    try:
        # with를 쓰면 cursor.close() 안써도 됨, cur = conn.cursor()
        with conn.cursor() as cur:
            # select문장 읽어와
            cur.execute("select code,sang,su,dan from sangdata order by code")
            
            # 커서로 읽어와
            rows = cur.fetchall()
        
        # 메세지도 받아와
        messages = list(get_flashed_messages())
        # 메세지와 select결과 들고가
        return render_template("list.html", rows=rows, messages=messages)
    
    # except pymysql.err.InterruptedError as e:
    #     err들을 구분구분 받을 수 있지만 한번에 하겠다
    except Exception as e:
        pass
    
    finally:
        conn.close()

""" add form 부르기 """
@app.get('/add/')
def add_form():
    messages = list(get_flashed_messages())
    return render_template("form_add.html", messages=messages) # 추가 form 호출

""" add 처리 """
@app.post("/add_save")
def add_save():
    # js에서 검사하고 넘어온 상태
    sang = (request.form.get("sang") or "").strip()
    su_raw = (request.form.get("su") or "").strip() # 넘어오는값은 : 문자열-"23"
    dan_raw = (request.form.get("dan") or "").strip()

    # 또 검사하기 
    # .isdigit() - 숫자인가
    if not sang or not su_raw.isdigit() or not dan_raw.isdigit():
        flash("sang은 필수, su/dan은 숫자만 허용")
        return redirect(url_for("add_form"))
    
    # 연산에 참여할거면 web에서 넘어오는 숫자데이터는 문자형으로 오기 때문에 형변환이 필수
    su = int(su_raw)
    dan = int(dan_raw)

    # 검수가 끝나면 이제야 다시 DB연결
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # code를 입력하지 않기 때문에 프로그램으로 입력할 것 
            # - 가장 큰값을 찾아서 다음순서로 넣음
            # DB가(orcle,...) 바뀌어도 변경하지 않아도 됨.
            cur.execute("select max(code) as max_code from sangdata") # mariadb에서 검증해보기
            row = cur.fetchone() # max값은 하나만 넘어와 단수
            
            # 가장 큰값이 없으면 None
            max_code = row["max_code"] if row else None
            next_code = (max_code + 1)if max_code is not None else 1

            # 추가하기 insert
            cur.execute("insert into sangdata(code,sang,su,dan) values (%s,%s,%s,%s)"
                        ,(next_code,sang,su,dan))
        # 오토커밋 꺼놨기 때문에 수동 커밋
        conn.commit()
        # 추가 후 목록으로 돌아가기
        return redirect(url_for("show_list"))

    except Exception as e:
        conn.rollback()
        flash("저장 실패 : " ,e)
        return redirect(url_for("add_form"))
    finally:
        conn.close()

""" edit form 호출 """
@app.get("/edit/<int:code>/")
def edit_form(code:int):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("select * from sangdata where code=%s",(code,))
            row = cur.fetchone()
        if not row:
                flash("해당 자료가 없습니다.")
                return redirect(url_for("show_list"))
        
        messages = list(get_flashed_messages())
        return render_template("form_edit.html",row=row, messages=messages)    
    finally:
        conn.close()

"""edit_save"""
@app.post("/edit/<int:code>/") # flask문법
def edit_save(code:int): # python문법
    # 수정처리
    sang = (request.form.get("sang") or "").strip()
    su_raw = (request.form.get("su") or "").strip() 
    dan_raw = (request.form.get("dan") or "").strip()

    if not sang or not su_raw.isdigit() or not dan_raw.isdigit():
        flash("sang은 필수, su/dan은 숫자만 허용")
        return redirect(url_for("edit_form", code=code))
    
    su = int(su_raw)
    dan = int(dan_raw)

    # 검수가 끝나면 DB연결
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 수정하기 update
            cur.execute("update sangdata set sang=%s,su=%s,dan=%s where code=%s"
                        ,(sang,su,dan,code))
        conn.commit()
        # 수정 후 목록으로 돌아가기
        return redirect(url_for("show_list"))
    except Exception as e:
        conn.rollback()
        flash("수정 실패 : " ,e)
        return redirect(url_for("edit_form"))
    finally:
        conn.close()

""" delete """
@app.post("/delete/<int:code>/")
def delete_row(code:int):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 삭제하기 delete
            cur.execute("delete from sangdata where code=%s",(code,))
        conn.commit()
        # 삭제 후 목록으로 돌아가기
        return redirect(url_for("show_list"))
    except Exception as e:
        conn.rollback()
        flash("삭제 실패 : " ,e)
        return redirect(url_for("show_list"))
    finally:
        conn.close()
    



if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")