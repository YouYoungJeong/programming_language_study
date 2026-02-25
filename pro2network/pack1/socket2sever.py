'''
socket server2 - 무한루프
'''
import socket
import sys

# 반복되는 값은 변수 지정해서 사용하는게 좋다.
# HOST = '127.0.0.1' # 이것만 가능해
HOST = '' # 누구든 가능해, 알아서 자기 컴퓨터의 아이피가 들어감.
PORT = 7788
seversock=socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

try:
    seversock.bind((HOST, PORT))
    seversock.listen(5)
    print('서버(무한 루핑) 서비스 중....')

    # 일반 웹서버는 다 이렇게 되어있음.-> 무한루핑
    # 받을때 decode, 넘길때 encode
    while True:
        # conn : Client
        conn, addr = seversock.accept()
        print('clinet info -', 'ip addr: ',addr[0],'\n','port num: ', addr[1])
        print(conn.recv(1024).decode()) # 수신 메세지 출력
        # 메세지 송신 to Client
        conn.send(('from server : ' + str(addr[1]) + '너도 잘지내라').encode('utf_8'))

except Exception as err:
    print('err :',err)
    # err나면 프로그램 종료
    sys.exit()
finally:
    conn.close()
    seversock.close()