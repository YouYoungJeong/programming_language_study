# 개인용 Database :  
# SQLite3 : 파이썬의 기본 모듈로 제공
# http://www.sqlite.org
# 모바일 기기, 임베디드 시스템 주로 사용.

import sqlite3
print(sqlite3.sqlite_version)

# local이라 db파일 하나만 만들어주면 바로 생성됨.
# conn = sqlite3.connect('exam.db')
# DB에 저장 하고 싶지 않음. RAM에만 DB저장 : 휘발성(1회용) 
conn = sqlite3.connect(':memory:')




# [예외처리 기본골격 : DB, network는 사용]
try:
    cur = conn.cursor(); # sql문을 사용하기위한 cursor()객체 생성

    # 테이블 생성
    cur.execute("create table if not exists friends(name text, phone text, addr text)")

    # 자료 입력
    cur.execute("insert into friends values('홍길동', '222-2222', '서초1동')")
    # 원래 받아오려면 ? 사용, 받아올데가 없어서 지금은 tuple로 받음.
    cur.execute("insert into friends values(?, ?, ?)",('신기해','333-3333','역삼 2동'))
    inputdatas = ('신기한','333-4444','역삼 2동')
    cur.execute("insert into friends values(?, ?, ?)",inputdatas)
    # 원본데이터에 들어가야함
    conn.commit()

    # 자료보기
    cur.execute("select * from friends")
    # 한 개의 레코드(행) 읽기
    # print(cur.fetchone())
    # 전체 다 읽기
    print(cur.fetchall()) # 읽은 내용을 다 소진하고
    print()

    # select문의 정석
    cur.execute("select name,phone,addr from friends") # 다시 읽음.
    # for문으로 불러오기.
    for r in cur:
        # print(r)
        print(r[0] + ' ' + r[1] + ' ' + r[2])

except Exception as e:
    print(f'err : {e}')
    # 잘못됬으면 롤백
    conn.rollback()
finally:
    # 열었으면 닫아라 필수
    conn.close()