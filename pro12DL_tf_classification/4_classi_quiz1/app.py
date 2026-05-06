# app.py
from flask import Flask, render_template, request, jsonify
from model import predict_diabetes

app = Flask(__name__)


@app.route("/")
def main():
    """
    메인 페이지 출력
    """
    return render_template("main.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Ajax 요청을 받아 model.py로 전달하고,
    예측 결과를 JSON으로 반환
    """
    try:
        data = request.get_json()

        result = predict_diabetes(data)

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)