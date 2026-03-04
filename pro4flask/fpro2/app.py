from flask import Flask, request, make_response
"""
request : 현재 들어온 http 요청 정보(파라미터, 폼, 헤더, 쿠키 ...)를 담는 객체 
make_response : 응답(response) 객체를 직접 만들어 반환할 때 사용하는 함수
출력을 html로 하고싶은데 지금은 너무 불편.
    html파일을 따로 만들어서 넘겨 -> jinja2
"""

app = Flask(__name__);

@app.route("/")
def home():
    return "<h2>홈 페이지</h2><p>/login으로 이동해 보세요</p>";

# 요청을 /login으로 하면 GET방식이야.
@app.route("/login", methods=['GET','POST']) # 요청 불가 허용하기

def login():
    #GET 요청 일때
    if request.method == 'GET':
        # 받아올 양이 많으면 """data""" 사용
        # method='post'로 넘김 - 유일하게 post방식으로 넘겨줄때 form tag안에!
        # 이방식 이외의 것들은 전부 GET방식
        return """
            <h2>로그인 페이지</h2>
            <form action="/login" method='post'>
                <input type="text" name="username" placeholder="사용자 이름 입력">
                <button type="submit">로그인</button>
            </form>
            <p>POST 요청시 username값을 서버가 받아 처리</p>
        """

    #POST 요청 일때
    elif request.method == 'POST':
        # 클라이언트가 request(요청)한값을 form tag로 받아왔어
        # js=> trim(), py => strip() : 앞뒤 공백 자르기
        user = request.form.get('username', '').strip();
        
        # user이름을 안주면
        if not user:
            return "사용자 이름을 입력하세요<br>" \
            "<a href='/login'>돌아가기</a>"

        # 정상 입력시 - 로그인 성공 메세지 출력
        message = f"""
                <h2>로그인 성공</h2>
                <p>안녕하세요 {user} 회원님. 서비스 마음껏 활용하세요</p>
                <a href="/">홈으로 돌아가기
        """
        # message를 return하고 성공하면 200을 반환
        return make_response(message, 200);
    else:
        return make_response("잘못된 요청입니다.", 405)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000);