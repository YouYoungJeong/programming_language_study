'''
socket server1 - 일회용 서버
'''
from socket import *

# 소켓 객체 생성
# AF_INET : 소켓종류 AF중 INET을 사용(특정 호스트를 찾아감), 
# SOCK_STREAM : 소켓 유형 : 신뢰성있게 주고받을 수 있는게 STREAM
seversock=socket(AF_INET, SOCK_STREAM) 

# 소켓을 특정 컴과 바인딩.(127.0.0.1) 8888포트로
seversock.bind(('127.0.0.1', 8888))

# 클라이언트와 연결 정보수 (1~5 까지 가능), 리스너 설정
seversock.listen(5)
print('서버 서비스 중....')

# 수동적으로 연결 받아들임.
# 능동적은 주식서버같이 특별한경우 사용 -> 계속 갱신해야하기때문에
conn, addr = seversock.accept()
print('client addr :', addr)
# client와 서로 주고받기하는 과정에서 받는걸 출력
print('from client message :', conn.recv(1024).decode())
conn.close()
seversock.close() 