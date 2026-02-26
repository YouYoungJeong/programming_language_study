'''server.py를 건들일때만 서버 재시작'''
# 한글 깨짐 방지
import sys
sys.stdout.reconfigure(encoding='utf-8')
# db연동, pickling
import MySQLdb
import pickle

# 피클링으로 DB계정 불러오기
with open("cgi-bin/mydb.dat", mode="rb")as obj:
    config = pickle.load(obj)


"""------------------------<HTML>------------------------------"""
print("Content-Type: text/html; charset=utf-8")
print()
print("<html><body><b>** 상품 정보 **</b><br/>")
print("<table border='1'>") # 표로 출력 tr:가로, td:세로
print("<tr><td>코드</td><td>상품명</td><td>수량</td><td>단가</td></tr>")

# SQL읽기
try:
    conn=MySQLdb.connect(**config)
    cursor = conn.cursor()

    cursor.execute("select * from sangdata order by code desc")
    datas = cursor.fetchall()
    for code,sang,su,dan in datas:
        print("""
            <tr>
                <td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td>
            </tr>
            """.format(code,sang,su,dan))

except Exception as err:
    print("err :",err)

finally:
    # 문장이 길어지면 \ 로 이어쓰기
    cursor.\
        close()
    conn.close()

print("</table>")
print("</html></body>")