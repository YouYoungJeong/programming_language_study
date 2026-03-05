from flask import Flask, render_template, request, \
    make_response, redirect, url_for
app = Flask(__name__);
COOKIE_AGE = 60 * 60 * 24 * 7 # 일주일간 보관
@app.get("/")
def home():
    return render_template("index.html")

@app.get("/login")
def loginfunc():
    # 로그인 할때마다 방문 횟수와 이름을 쿠키에 등록
    name = request.cookies.get("name")
    visits = request.cookies.get("visits")
    
    if name:
        # 웹에서 넘어오는건 전부 문자열이야 형변환 + 방문횟수
        # 방문 횟수가 없으면 0
        # 이름이 있으면 방문 횟수 1 증가
        visits = int(visits or "0") + 1
        msg = f"안녕하세요. {name}님 {visits}번째 방문을 환영합니다."
    else:
        visits = None
        msg = "이름을 입력하면 방문 횟수를 쿠키로 기억합니다."
    resp = make_response(render_template("login.html", msg=msg, name=name, visits=visits))
    
    # name이 있는 상태(로그인 상태)면 visits 쿠키 갱신
    if name: # 변수명과 값이 같을때 => visits=visits == visits
        resp.set_cookie("visits", str(visits), max_age=COOKIE_AGE, samesite="Lax")
    
    return resp

@app.post("/login")
def loginfunc2():
    name = (request.form.get("name") or "").strip()
    # 쿠키를 만들어서 "loginfunc"요청을 redirect객체로 만들어서 response함
    resp = make_response(redirect(url_for("loginfunc")))
    # 쿠키 생성
    resp.set_cookie("name", name, max_age=COOKIE_AGE, samesite="Lax")
    resp.set_cookie("visits", "0", max_age=COOKIE_AGE, samesite="Lax")
    # 쿠키값 loginfunc에 반환
    return resp

@app.post("/logout")
def logoutfunc():
    # 쿠키 삭제 후 /login(get)으로 이동
    resp = make_response(redirect(url_for("loginfunc")))
    resp.delete_cookie("name")
    resp.delete_cookie("visits")
    return resp
if __name__==("__main__"):
    app.run(debug=True, host="0.0.0.0")