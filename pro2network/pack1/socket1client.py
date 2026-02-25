'''
socket client1 - 일회용 서버
'''
from socket import *

clientsock=socket(AF_INET, SOCK_STREAM)

# 바인딩과 능동적으로 연결을 시도함(튜플)
clientsock.connect(('127.0.0.1', 8888))
# 연결후 보내 encode로 보내서 decode로 풀어서 확인
clientsock.send('안녕 반가워'.encode(encoding='utf_8', errors='strict'))
clientsock.close()
clientsock.close()