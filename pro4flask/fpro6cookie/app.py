''' 
Cookie
    브라우저에 저장되는 작은 키-값 데이터이고, 
    서버가 클라이언트와 연결 유지하는것 처럼 할 수 있다.
    서버가 설정 -> 브라우저가 저장 -> 다음 요청 부터 브라우저가 자동으로 함께 전송
'''
""" 
[모듈 역할]
Flask : 웹 어플리케이션을 만드는 클래스
render_template_string : 문자열로 작성한 jinja템플릿을 렌더링해 HTML로 반환할때 사용
request : 클라이언트의 http요청을 받아올때 사용
make_response : 응답객체를 직접 만듦. 쿠키, 헤더등 설정할 때도 사용
redirect : 다른 URL로 이동할때 사용
url_for : *route함수 이름으로 URL을 안전하게 생성하는 함수
"""
'''------------------------------------------------------'''
from flask import Flask, render_template_string, request, \
    make_response, redirect, url_for
app = Flask(__name__)

'''HTML만들기
    render_template_string모듈로 html string불러오기 가능'''
HOME_HTML="""
    <h2>Flask</h2>
    <form action='/set_cookie' method='post'>
        쿠키 값 : <input type='text' name='name' placeholder='예:hong'>
        <button type='submit'>쿠키 저장</button>
    </form>
    <p>
        <a href='/read_cookie'>쿠키 읽기</a>
        <a href='/delete_cookie'>쿠키 삭제</a>
    </p>
"""

@app.get("/")
def home():
    return render_template_string(HOME_HTML)

@app.post("/set_cookie")
def set_cookie():
    # 쿠키 저장
    name = request.form.get("name", "anonymous")

    # 클라이언트에 쿠키를 심으려면 응답 객체가 필요
    """
    먼저 'read_cookie페이지로 이동해라' 라는 redirect객체를 만들고 
    -> 그 응답에 따라 쿠키를 추가한 뒤 브라우저에 돌려줌 
    redirect : 클라이언트를 통해서 서버로 요청
    (redirect("read_cookie"))
    """
    resp = make_response(redirect(url_for("read_cookie")))
    # 브라우저에 쿠키 저장 설정값
    resp.set_cookie(
        key="name",         #  쿠키 이름
        value=name,         # 사용자가 입력한 쿠키값
        max_age = 60 * 5,   # 유효시간 - 5분뒤 만료, 보통은 1년유지 시킴.   
        httponly=True,      #JS에서 document.cookie로 접근 불가
        samesite="Lax"      #CSRF 공격 방지용, 적당한 규제
    )
    return resp             # 쿠키가 포함된 응답을 브라우저로 반환
    # 응답을 브라우저로 반환하는데 브라우저는 쿠키를 저장하고 
    # redirect요청에 따라 read_cookie로 다시 요청함
@app.get("/read_cookie")
def read_cookie():
    # 브라우저가 요청에 실어 보낸 모든 쿠키 중에서 내 서버가 만든 name쿠키를 꺼냄.
    # 없으면 None을 반환 - 첫방문,만료,삭제된 경우
    # 클라이언트 컴퓨터에 쿠키가 잔뜩 있어서 cookies, 
    # 원래 name은 단순하지 않음.
    name = request.cookies.get("name")

    # 읽은 쿠키 html로 출력하기
    return f"""
        <h3>쿠키 읽기</h3>
        <p>name 쿠키 값 : {name} </p>
        <a href='/'>홈페이지</a>
    """
    # 쿠키 확인하기 : 검사 - application - Cookies

"""쿠키 삭제하기"""
@app.get("/delete_cookie")
def delete_cookie():
    # 쿠키 삭제 후 홈(/)으로 이동하기 위한 redirect 응답을 만듦
    resp = make_response(redirect(url_for("home")))
    resp.delete_cookie("name")
    return resp

if __name__==("__main__"):
    app.run(debug=True, host="0.0.0.0")