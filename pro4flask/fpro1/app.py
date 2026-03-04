"""
Python Application Server : Python프로그램 코드를 실행해서 요청을 처리하는 서버
Flask 기본 서버는 실무용 아님. 
    개발용, 학습용 : Ligth-weight server
    실무용 서버(WSGI) : gunicorn(리눅스용), waitress(윈도우도 가능), nginx ...
"""
# pip install flask - 플라스크 모듈 터미널 설치
from flask import Flask # 웹 서버(Application Server) 생성에 필요.

# waitress 서버를 사용한다면 :pip install waitress
from waitress import serve

# __name__ : 현재 모듈의 이름 을 생성자에 넣어서 flask 객체 생성
app = Flask(__name__); 


# @app.route("/"):URL 매핑(라우팅) - 클라이언트요청이 '/'일때 abc()함수 수행
@app.route("/")
# 클라이언트 요청을 처리하는 함수
def abc():
    return "<h1>안녕하세요</h1> 반가워요"; # 클라이언트 브라우저에게 반환


@app.route("/about") # url:http://127.0.0.1:5000/about 실행
def about():
    return "플라스크를 소개하자면 음 만세~"

@app.route("/user/<name>")  # url에 변수를 담아 요청 : http://127.0.0.1:5000/user/신기해
def user(name):
    return f"내 친구 {name}"

if __name__ == "__main__":
    # Flask 기본 서버 호출 - 학습용
    # app.run();
    # app.run(debug=True, host="0.0.0.0", port=5000); 

    # waitress 실무용 서버 호출 - http://127.0.0.1:8000
    # 다른사람이 들어올 수 있게 사용하려면 이걸 사용
    print("웹서버 서비스 시작 ....")
    serve(app=app,host="0.0.0.0", port=8000)
"""
개발할때만 True,-> Debugger is active! 
Restarting with watchdog (windowsapi) - 코드 수정을 자동 감지하는 라이브러리
배포는 False.
0.0.0.0 외부에서 자유롭게 접속이 가능한 host번호
port번호도 마음대로 가능, flask는 5000번 사용
"""


"""
터미널 python app.py - 서버 실행
Debug mode : 자동갱신 -  off면 수정 후 서버 껏다켜.
GET / HTTP/1.1" 200 - get방식, 서버의 상태가 멀쩡하니까 200을 반환
"""

"""
흐름
1. 클라우드 요청
2. route
3. 함수 수행
4. 결과 return
"""