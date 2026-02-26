'''
SimpleHTTPRequestHandler 는 python을 쓸 수 없다
CGIHTTPRequestHandler : SimpleHTTPRequestHandler의 확장 클래스
get, post 모두 지원 가능

- CGI(Common Gateway Interface) : 
    웹서버와 외부 프로그램 사이에서 정보를 주고받는 방법이나 규약


'''
from http.server import HTTPServer, CGIHTTPRequestHandler

PORT = 8888

"""
HTTPServer는 CGIHTTPRequestHandler를 받아야함. 상속
받으면서 '/cgi-bin'를 같이 보냄.
"""
class Handler(CGIHTTPRequestHandler):
    # 받을 파일이 여러개면 이런식으로 디렉토리를 <- list로 받았다.
    cgi_directories=['/cgi-bin']


def run():
    serv = HTTPServer(('127.0.0.1', PORT), Handler)
    
    print('웹서비스 진행중 ...')
    try:
        # 무한
        serv.serve_forever()
    except Exception as err:
        print('서버 종료')
    finally:
        serv.server_close()

if __name__ =='__main__':
    run()