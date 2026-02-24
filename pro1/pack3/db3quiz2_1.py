'''
문2-1) 직원번호와 직원명을 입력(로그인)하여 성공하면 아래의 내용 출력
해당 직원이 근무하는 부서 내의 직원 전부를 직급별 오름차순우로 출력. 직급이 같으면 이름별 오름차순한다.
직원번호 입력 : _______
직원명 입력 : _______
직원번호 직원명 부서명 부서전화 직급 성별
1 홍길동 총무부 111-1111 이사 남
...
직원 수 :

이어서 로그인한 해당 직원이 관리하는 고객 자료도 출력한다.
고객번호 고객명 고객전화 나이

1 사오정 555-5555 34

관리 고객 수 :
'''
import MySQLdb

config = {
    'host':'127.0.0.1',
    'user' : 'root',
    'password':'123',
    'database':'test',
    'port':3306, 
    'charset':'utf8'
}
def quiz2_1():
    try:
        conn=MySQLdb.connect(**config)
        cursor = conn.cursor() 
        
        #직원 데이터 받기
        jik_no = input('직원번호 입력 :')
        jik_name = input('직원이름 입력 :')

        # 직원 로그인
        sql1 = """
            select  jikwonno 직원번호, jikwonname 직원명,busername 부서명, busertel 부서전화,
            jikwonjik 직급, jikwongen 성별 from 
            jikwon left outer join buser on busernum=buserno
            where busernum = (select busernum from jikwon where jikwonno={0})
            order by jikwonjik, jikwonname
            """.format(jik_no)
        # print(sql1)
        cursor.execute(sql1)
        selectsql1 = cursor.fetchall()
        for (jikwonno,jikwonname,busername,busertel,jikwonjik,jikwongen) in selectsql1:
            print(jikwonno,jikwonname,busername,busertel,jikwonjik,jikwongen)
        print('직원 수 :', len(selectsql1))        

        # 로그인한 직원 담당 고객 출력
        sql2 = """
                select gogekno 고객번호, gogekname 고객명, gogektel 고객전화, 
                TIMESTAMPDIFF(YEAR,(SUBSTR(gogekjumin,1,6)),NOW()) 나이 
                from gogek left outer join jikwon on jikwonno=gogekdamsano
                where jikwonno={0}
                """.format(jik_no)
        # print(sql2)
        cursor.execute(sql2)
        selectsql2 = cursor.fetchall()
        if len(selectsql2)==0:
            print('담당 고객이 존재하지 않습니다.')
        for (gogekno,gogekname,gogektel,age) in selectsql2:
            print(gogekno,gogekname,gogektel,age)
        print('고객 수 :', len(selectsql2))


    except Exception as e:
        print(f'err : {e}')
        conn.rollback() # select 에는 필요 X
    
    finally: 
        cursor.close()
        conn.close()



if __name__=='__main__':
    quiz2_1()