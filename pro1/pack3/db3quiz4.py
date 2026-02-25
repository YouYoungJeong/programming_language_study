'''
문4)직원별 관리 고객 수 출력 (관리 고객이 없으면 출력에서 제외)
직원번호 직원명  관리 고객 수
1         홍길동    3
2          한송이  1
'''
import MySQLdb
import pickle

with open('mydb.dat', mode='rb') as obj:
    config = pickle.load(obj)

def quzi4():
    try:
        conn=MySQLdb.connect(**config)
        cursor = conn.cursor() 
        sql = """
            select jikwonno, jikwonname, count(gogekdamsano)
            from gogek left outer join jikwon on gogekdamsano=jikwonno
            group by jikwonno
            """
        # == jikwon inner join gogek 
        cursor.execute(sql)
        selectsql = cursor.fetchall()
        print('직원번호 직원명\t관리고객수')
        for (jikwonno,jikwonname,count) in selectsql:            
            print("{}\t{}\t{}".format(jikwonno,jikwonname,count))
        
    except Exception as e:
        print(f'err : {e}')
        conn.rollback() # select 에는 필요 X
    
    finally: 
        cursor.close()
        conn.close()

if __name__=='__main__':
    quzi4()