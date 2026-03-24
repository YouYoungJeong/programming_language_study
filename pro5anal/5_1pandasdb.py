"""
local db 연동 후 DataFrame에 자료 저장
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

print('-'*15,'rows를 DataFrame에 저장','-'*15)
import pandas as pd
df1 = pd.DataFrame(rows, columns=['product','maker','weight','price'])
print(df1)
print(df1.describe())

cursor.close()
conn.close()