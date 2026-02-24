'''
문1) jikwon 테이블 자료 출력
키보드로부터 부서번호를 입력받아, 해당 부서에 직원 자료 출력

부서번호 입력 : _______
직원번호 직원명 근무지역 직급
1 홍길동 서울 이사
...
인원 수 :
'''
import MySQLdb
import pickle


'''
config = {
    'host':'127.0.0.1',
    'user' : 'root',
    'password':'123',
    'database':'test',
    'port':3306, 
    'charset':'utf8'
}
'''
# ☆ config정보를 따로 피클링으로 불러오기
with open('mydb.dat', mode='rb') as obj:
    config = pickle.load(obj)

def chulbal():
    try:
        conn=MySQLdb.connect(**config)
        cursor = conn.cursor()
        
        # 부서 번호 받기
        bu_no = input('부서 번호 입력:')
        # print(bu_no) # input값 확인하기.

        # sql문장 생성 -> prompt에서 실행 한번 확인
        sql = """
            select jikwonno as 직원번호, jikwonname as 직원명,
            buserloc as 근무지역, jikwonjik as 직급
            from jikwon 
            inner join buser on busernum=buserno
            where busernum={0}
            """.format(bu_no)
        # print(sql) # 출력받은 값 들어오면서 sql문장 완성됐는지 확인.
        
        cursor.execute(sql) # <- 설명 할 줄 알아야함. 한줄씩 읽어옴.
        
        # 읽는 방법1
        datas = cursor.fetchall()
        # print(datas)
        
        if len(datas)==0:
            print(bu_no + '번 부서는 없습니다.')
            return  # 함수 종료 , sys.exut(0) : 응용프로그램 강제 종료 -> 지금은 둘다 써도 똑같아
        
        for jikwonno, jikwonname, buserloc, jikwonjik in datas:
            print(jikwonno, jikwonname, buserloc, jikwonjik) 

        # SQL에서 count할 필요 없이 datas로 받아오는 건수를 세면 됨.
        print('인원수 :',str(len(datas))) 

    except Exception as e:
        print(f'err : {e}')





    finally: 
        cursor.close()
        conn.close()









if __name__=='__main__':
    chulbal()