"""
local DB 연동 후 DataFrame에 자료 저장 - sqlite3사용
"""
# try문 생략
import sqlite3
sql = "create table if not exists extab(product varchar(10), maker varchar(10), weight real, price integer)"
conn = sqlite3.connect(':memory:') # 실행하는 동안만 존재, 실험용
conn.execute(sql)
conn.commit()

data = [('mouse', 'samsung', 12.5, 5000), ('keyboard', 'lg', 52.5, 35000)]
isql = 'insert into extab values(?,?,?,?)'

# 데이터를 여러개 줄때 executemany
conn.executemany(isql, data)

# 데이터를 하나만 줄때 execute
data1 = ('pen','abc','5.0','1200')
conn.execute(isql, data1)
conn.commit()
cursor = conn.execute("select * from extab")
rows = cursor.fetchall()
for a in rows:
    print(a)
    for j in a:
        print(j)
print()

print('-'*15,'rows를 DataFrame에 저장1 - fetchall','-'*15)
import pandas as pd
df1 = pd.DataFrame(rows, columns=['product','maker','weight','price'])
print(df1)
print(df1.describe())
print()

print('-'*15,'rows를 DataFrame에 저장2 - read_sql()','-'*15)
df2 = pd.read_sql("select * from extab", conn) # sql문 연결객체 conn
print(df2)
print(pd.read_sql('select count(*) as 건수 from extab', conn))
print()

print('-'*15,'DataFrame의 자료를 테이블에 저장(insert) - to_sql','-'*15)
data = {
    'product':['연필','볼펜','지우개'],
    'maker':['모나미','모나미','모나미'],
    'weight':[2.3, 3.0, 5.0],
    'price':(1000, 2000, 500)
}
frame = pd.DataFrame(data)
# print(frame)
# auto commit
frame.to_sql('extab', conn, if_exists='append', index=False) #if_exists있으면 추가(append)
df3 = pd.read_sql("select * from extab", conn) # sql문 연결객체 conn
print(df3)



cursor.close()
conn.close()