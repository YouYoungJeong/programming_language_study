ss = "python 자료 출력"
ss2 = 123 + 5
# 서버에서 인터프리터로 해석하고 클라이언트에는 html로 바꿔서 전송
# 클라이언트 브라우저는 받은 html을 파싱함.
# print : 개발자가 자신의 컴 표준출력장치로 값 확인
print(ss)

# 클라이언트 브라우저로 출력(Content-Type : 첫글자 대문자)
print("Content-Type:text/html;charset:utf-8\n")
# html
print("<html>")
print("<body>")

print("<b>안녕. 파이썬 모듈로 작성한 문서야</b><br/>")
# 파이썬 변수 출력(튜플로 주는게 필수인가?)
print("파이썬 변수 값1 : %s"%(ss,))
print("<br/>파이썬 변수 값2 : %d"%(ss2,))

print("</body></html>")