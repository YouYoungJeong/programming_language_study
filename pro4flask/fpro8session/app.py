from flask import Flask, render_template, request, session\
    , make_response, redirect, url_for
"""
파이썬세션(Session)
    웹에서 사용자 정보를 서버에 저장하는 기능을 말함 (쿠키를 통해 세션 운영)
    일정 시간 동안 동일 사용자(브라우저)와 
        일련의 요청을 하나의 상태로 보고 그상태를 유지시키는 기술 
    쿠키에 비해 상대적으로 안전
"""
# 날짜/시간을 가감(+-)해서 기간 설정하기에 효과적
from datetime import timedelta 

# 실습 : 사용자가 os를 선택하면 세션에 저장하고 읽기
app = Flask(__name__)

# Flask는 세션 사용을 위해 secret_key(비밀키) 설정이 필요
app.secret_key= "abcdef123456" # 위조 방지용 비밀키값 - 세션이 진짜인지 가짜인지
"""참고 키값 자동생성 터미널창에서 
> python -c "import secrets; print(secrets.token_hex(32))"
> "b8bfb0dcf2381c5d87f7ed4556c22f64fcf67efae29f508e50f2958a9fc86571" 이런식으로 사용
"""

app.permanent_session_lifetime = timedelta(seconds=5) # 세션 만료시간 상대적인 5초 설정 
"""설정을 안하면 : (defualt=30m) , 시간은 서버에서 주는 나름
상대적인 5초 -> 활동하는 순간에는 계속해서 갱신. 활동을 멈추면 그때 5초가 지나면 만료 
"""

@app.get("/")
def home():
    return render_template("main.html")

@app.route("/setos")
def setos():
    favorite_os = request.args.get("favorite_os")

    if favorite_os:
        session.permanent = True # 세션 만료 시간 적용
        session["f_os"] = favorite_os # "f_os"라는 키로 특정 값 저장 - 세션에 저장
        return redirect(url_for("showos"))
    else:
        return render_template("setos.html")
    
@app.route("/showos")
def showos():
    context = {}

    if "f_os" in session:
        context["f_os"] = session["f_os"]
        context["message"] = f"당신이 선택한 운영체제는 '{session['f_os']}'"
    else:
        context["f_os"] = None
        context["message"] = "운영체제를 선택하지 않았거나, 세션이 만료됨"

    return render_template("showos.html", context=context) # 묶음형 자료를 넘길 수 있다.


if __name__==("__main__"):
    app.run(debug=True, host="0.0.0.0")