# html이 가능한서버 = http서버
# https://cafe.daum.net/flowlife/RUrO/144
# Flask가기전에 직접 단순한 HTTPServer 구축 - 기본적인 socket연결 관리
# SimpleHTTPRequestHandler 는 python은 해석을 못함.
# tcpip를 class로 만든게 소켓

from http.server import SimpleHTTPRequestHandler, HTTPServer

# 상수값을 줄때 대문자로 쓰자(보이지 않는 약속)
PORT = 7777

# SimpleHTTPRequestHandler : 클라이언트의 get 요청에 대해 문서를 읽어 클라이언트로 전송하는 역할을 함.(요청을 처리)
handler = SimpleHTTPRequestHandler

# HTTPServer 객체 생성
# serv = HTTPServer(('192.168.0.114',PORT), handler)
serv = HTTPServer(('127.0.0.1',PORT), handler)

print('웹 서비스 시작...')
# 웹서비스 무한 루핑
serv.serve_forever()

# 웹서버 만들어서
# html 파일로 출력물 만들기
# 웹에서 내 컴퓨터 포트로 들어가기 : http://127.0.0.1:7777
# 웹에서 소스보면 html소스가 보임.
# http://localhost:7777/abc.html : localhost는 loopback이라고 함 = 127.0.0.1
# <!DOCTYPE html>  <- 이게 보이면 html 최신버전