import numpy as np
import pandas as pd

data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# print(data[::-1,::-1])

df = pd.DataFrame(np.arange(12).reshape(4,3), columns=['강남','강북','서초'], index=['1월','2월','3월','4월'])
# print(df)

from pandas import DataFrame
frame = DataFrame({'bun':[1,2,3,4], 'irum':['aa','bb','cc','dd']}, index=['a','b', 'c','d'])
# print(frame.T)
frame2 = frame.drop("d")
# print(frame2)
df = pd.read_csv("ex1.csv", names=[ 'a', 'b', 'c', 'd'])
# print(df)
data = {
    'juso':['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon':[23, 25, 15]
}
df = pd.DataFrame(data)
# print(df)
results = pd.Series([x.split()[0] for x in df.juso])
# print(results)

x = np.array([1,2,3,4,5])
y = np.arange(1, 4).reshape(3, 1)
# print(x + y)

import MySQLdb
import numpy as np
import sys
def main():
    CONFIG = {"host": "127.0.0.1", "user": "root", "passwd": "123", "db": "test", "port": 3306, "charset": "utf8" }
    sql = """
        select count(*) as 직원수, avg(jikwonpay) as 급여평균, std(jikwonpay) as 표준편차
        from jikwon left outer join gogek 
        on jikwon.jikwonno = gogek.gogekdamsano where gogekdamsano is null
        """
    conn = MySQLdb.connect(**CONFIG)
    cursor = conn.cursor()
    cursor.execute(sql)
    jikdf = pd.DataFrame(cursor.fetchall(), columns=['직원수', '급여평균','표준편차'])
    # print(jikdf.head())
if __name__ == "__main__":
    main()

data = pd.DataFrame(np.random.randn(36).reshape(9,4), columns=['가격1', '가격2', '가격3', '가격4'])
data_mean = data.mean(axis=0)
# print(data_mean)

data = {"a": [80, 90, 70, 30], "b": [90, 70, 60, 40], "c": [90, 60, 80, 70]}
data_df= pd.DataFrame(data)
data_df.columns=["국어", "영어", "수학"]
# print(data_df['수학'])
# print(data_df['수학'].std())
# print(data_df[['국어','영어']])

import matplotlib.pyplot as plt
data = np.random.randn(1000)
print(data.mean(), data.std())
plt.hist(data, bins=20, alpha=0.7)
plt.title('good')
# plt.show()

df = pd.read_csv('sales_data.csv')
df_piv = df.pivot_table(
    index=['날짜','제품'],
    values='판매수량',
    aggfunc='sum'
).unstack().reset_index()
print(df_piv)