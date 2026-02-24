'''
문3) 성별 직원 현황 출력 : 성별(남/여) 단위로 직원 수와 평균 급여 출력
성별 직원수 평균급여
남     3     8500
여     2     7800
'''
import MySQLdb
import pickle

with open('mydb.dat', mode='rb') as obj:
    config = pickle.load(obj)

def quzi3():
    try:
        conn=MySQLdb.connect(**config)
        cursor = conn.cursor() 
        sql = """
            select nvl(jikwongen, '미입력') 성별, count(jikwonno), 
            round(avg(jikwonpay))
            from jikwon group by jikwongen
            """
        cursor.execute(sql)
        selectsql = cursor.fetchall()
        print('성별\t직원수\t평균급여\t')
        for (gen, count, avg) in selectsql:            
            print("{}\t{}\t{}".format(gen,count,avg))
        
    
    except Exception as e:
        print(f'err : {e}')
        conn.rollback() # select 에는 필요 X
    
    finally: 
        cursor.close()
        conn.close()

if __name__=='__main__':
    quzi3()