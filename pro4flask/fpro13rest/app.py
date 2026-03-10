from flask import Flask, render_template, request, jsonify
# jsonify : Json모양을 만들어줌

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

@app.get('/api/friend')
def api_friendFunc():
    name = request.args.get("name","").strip()
    age_str = request.args.get("age","").strip()
    # 입력 검증후 값 반환
    if not name:
        return jsonify({"ok":False, "error":"name is required"}), 400 # http상태 코드도 같이 넘김
    if not age_str.isdigit():
        return jsonify({"ok":False, "error":"age is required"}), 400

    age = int(age_str)
    age_group = f"{(age // 10) * 10}대" # 정수나누기 * 10 = 나눈몫*10 대
    
    # 리턴값 json type
    return jsonify({
        "ok":True,
        "name":name,
        "age":age,
        "age_group":age_group,
        "message":f"{name}님은 {age}살 {age_group}입니다."
    })


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")