from flask import Flask, render_template, request, jsonify
import pymysql

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("main.html")

@app.get("/legacy")
def legacyf():
    pass
@app.get("/asuyc")
def asuycf():
    pass

@app.get("/fetch")
def fetchf():
    return render_template("fetchshow2.html")

@app.get("/axios")
def axiosf():
    return render_template("axiosshow3.html")

# sangdata 가져오기
@app.get("/api/sangdata")
def sangdata():
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="123",
        database="test",
        charset="utf8"
    )
    cur = conn.cursor()
    cur.execute("select * from sangdata")
    columns = [col[0] for col in cur.description]
    rows = cur.fetchall()
    result = [dict(zip(columns, row)) for row in rows]
    print(result)
    cur.close()
    conn.close()
    return jsonify(result)


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")