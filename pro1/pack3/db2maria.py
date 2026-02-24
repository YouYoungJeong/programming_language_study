# 원격 데이터베이스 연동 프로그래밍
# MariaDB : driver file 설치 후 사용
# pip install mysqlclient - 여러개 있는데 제일 사용하기 쉬움.
# 설치후 C:\Users\acorn\anaconda3\envs\myproject\Lib\site-packages\MySQLdb 
# 있는지 확인
# cmd에서 ipconfig치면 -> 
# IPv4 주소 . . . . . . . . . : 192.168.0.182
import MySQLdb
'''
conn = MySQLdb.connect(
    host='127.0.0.1',
    user = 'root',
    password='123',
    database='test',
    port=3306)
print(conn)
conn.close()
'''

# sangdata 자료 CRUD

# 1.DB 연결 정보 dict에 저장하기, 순서 상관없음.
config = {
    'host':'127.0.0.1',
    'user' : 'root',
    'password':'123',
    'database':'test',
    'port':3306, # int로 적어야해.
    'charset':'utf8'
}

def myFunc():
    try:
        conn=MySQLdb.connect(**config) # dict type 받을때 **
        cursor = conn.cursor()

        '''-------------------자료 추가---------------------'''
        # # 추가방법 1
        # # 문자는 ''사용, 숫자는 ''써도되고 안써도됨. 전체가 문자열로 들어가기때문에 제약조건에서 알아서 들어감.
        # isql = "insert into sangdata(code,sang,su,dan) values(5,'신상1',5,'7800')"
        # # local에서만 insert된 상태.트랜잭션 시작.
        # cursor.execute(isql) # code가 pk라 한번만 실행해야함. 
        # # 트랜잭션 종료
        # conn.commit()

        # # 추가방법 2
        # isql = "insert into sangdata values(%s,%s,%s,%s)"
        # # sql_data=(6,'신상2',11,5000)
        # sql_data=6,'신상2',11,5000 # 튜플형식은 괄호 없어도 됨.
        # cursor.execute(isql, sql_data)
        # conn.commit()

        '''------------------자료 수정--------------------'''
        # PK는 수정 대상 제외, 순서는 바꿔도 됨.
        # 수정 방법1
        # usql = "update sangdata set sang=%s,su=%s,dan=%s where code=%s"
        # sql_data=('물티슈',66,1000,5)
        # cursor.execute(usql, sql_data)
        # conn.commit()

        # 수정 방법2 : *return값 출력하기 
        # usql = "update sangdata set sang=%s,su=%s,dan=%s where code=%s"
        # sql_data=('콜라',77,1000,5)
        # cou = cursor.execute(usql, sql_data)
        # print('수정 건수 : ',cou)
        # conn.commit()

        '''------------------자료 삭제--------------------'''
        # SQL은 전체가 문자열
        code = '6'

        # 값 받는 형식1
        # 문자열 더하기로 SQL완성 비권장 
        # - secure coding 가이드라인 위배(SQL 인젝션(SQL Injection)공격 받을 수 있다.)
        # secure coding 가이드라인에 위해하면 리젝당함.
        # dsql = "delete from sangdata where code=" + code 

        # 값 받는 형식2
        # dsql = "delete from sangdata where code=%s"
        # # code가 그냥 str이라 tuple형태"(code,)"로 넣어줘야함.
        # cursor.execute(dsql,(code,)) 
        # conn.commit()

        # 값 받는 형식3
        dsql = "delete from sangdata where code='{0}'".format(code)
        cou = cursor.execute(dsql) # 삭제후return값 받기(0 또는 1이상의 값)
        if cou !=0:
            print('삭제 성공') 
        else:
            print('삭제 실패')
        conn.commit()

        '''------------------자료 읽기--------------------'''
        #(DB서버의 자료가 내 컴퓨터의 RAM(주기억장치)로 들어와있고 커서 장치로 읽어옴)
        sql="select code,sang,su,dan from sangdata"
        cursor.execute(sql)

        # 출력방법1 # <-- 왜 for문으로 출력하는지?
        for data in cursor.fetchall(): 
            # print(data) # tuple형태로 출력
            print('%s %s %s %s'%data)
        print()
        
        # 출력방법2
        cursor.execute(sql)
        for r in cursor:
            print(r[0], r[1], r[2], r[3])
        print()

        #출력방법3 - 가독성이 좋다.
        cursor.execute(sql)
        for (code, sang, su, dan) in cursor:
            print(code, sang, su, dan)
        print()

        #출력방법3_1 -> 튜플형태라 () 없어도 됨?
        cursor.execute(sql)
        for code, sang, su, dan in cursor: # 그냥 변수명임 마음대로 써도됨.들어오는 순서대로 맵핑.
            print(code, sang, su, dan)
        print()

        #출력방법4
        cursor.execute(sql)
        for (a, b, 수량, 단가) in cursor:
            print(a, b, 수량, 단가)

    except Exception as e:
        print(f'err : {e}')
        conn.rollback()

    # finally : Err가 나든 안나든 반드시 수행함.
    finally: 
        # 만든거 역순으로 close
        cursor.close()
        conn.close()


# main모듈이면 실행시켜    
if __name__=='__main__':
    myFunc()