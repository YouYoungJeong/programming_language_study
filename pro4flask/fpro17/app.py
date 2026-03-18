from flask import Flask, render_template, request, jsonify
from db import get_connFunc

app= Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

# 전체 직원 조회
@app.get("/api/jikwon")
def jikwon_list():
    sql = """
    select jikwonno, jikwonname, busername, jikwonjik, 
    jikwonpay, year(jikwonibsail) as ibsayear
    from jikwon
    inner join buser on jikwon.busername = buser.buserno
    order by jikwonno
    """

    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return jsonify({"ok": True,"data":rows})

# 직원 1명 조회
@app.get("/api/jikwon/<int:on>")
def jikwon_one(no):
    sql = """
    select jikwonno, jikwonname, busername, jikwonjik, 
    jikwonpay, year(jikwonibsail) as ibsayear
    from jikwon
    inner join buser on jikwon.busername = buser.buserno
    where jikwonno = %s
    """

    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (no,))
            rows = cur.fetchone()
    return jsonify({"ok": True,"data":rows})

# 전체 부서 조회
@app.get("/api/buser")
def jikwon_list():
    sql = """
    select * from buser order by buserno """

    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return jsonify({"ok": True,"data":rows})


# 특정 부서 직원 조회
@app.get("/api/jikwon/<int:bon>/jikwon")
def buser_jikwon(bno):
    sql = """
    select jikwonno, jikwonname, jikwonjik, 
    jikwonpay, year(jikwonibsail) as ibsayear
    from jikwon where busernnum = %s
    """
    with get_connFunc() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return jsonify({"ok": True,"data":rows})


if __name__ == "__main__":
    app.run(debug=True)