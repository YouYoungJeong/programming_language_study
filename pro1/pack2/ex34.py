'''
파일 처리
- 꼭 try 사용 (file, db, network....)
open()
close()
'''
import os

try:
    print(os.getcwd()) #현재 경로 C:\work\projects\pro1\pack2

    print("파일 읽기, mode='r'")
    # f1 = open(os.getcwd() + r"\ftext.txt" ,encoding='utf-8')
    # f1 = open(r'C:\work\projects\pro1\pack2\ftext.txt' ,encoding='utf-8')
    # f1 = open("ftext.txt",encoding= 'utf-8')
    f1 = open("ftext.txt", mode='r' ,encoding= 'utf-8') # mod = 'r', 'w', 'a', 'b'...이 있다 <= 정리 
    print(f1)
    print(f1.read())
    f1.close

    print("파일 저장, mode='w' ")
    f2 = open('ftext2.txt', mode='w', encoding='utf-8')
    f2.write('내 친구들\n')
    f2.write('홍길동, 한국인')
    f2.close()
    print('파일 저장 성공')

    print("파일 내용 추가, mode='a'")
    f3 = open('ftext2.txt', mode='a', encoding='utf-8')
    f3.write('\n사오정')
    f3.write('\n저팔계')
    f3.write('\n손오공')
    f3.close()
    print('파일 추가 성공')


    f4 = open("ftext2.txt", mode='r' ,encoding= 'utf-8')
    print(f4.read())
    f4.close()
except Exception as e:
    print('파일 처리 오류', e)
