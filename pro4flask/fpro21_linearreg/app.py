from flask import Flask, render_template, request, redirect, jsonify
import pandas as pd
from db import fetchall_avgpay
from analysis import linearFunc


app = Flask(__name__)
@app.get("/")
def main():
    data = fetchall_avgpay()
    # print(data)
    df = pd.DataFrame(data)
    df = df.astype({"평균연봉":int})

    return render_template('main.html', result=df.to_html(index=False))

@app.get('/predict')
def analdata():
    # main에서 값 받아오기
    year = request.args.get('data')

    # main에서 받아온값 보내서 R² , 회귀식,  연봉 예측값 받아오기
    r_scoremsg, sik, pay_pred = linearFunc(year)
    
    # ajax -> json으로 보내기
    return jsonify({
        'r_scoremsg': r_scoremsg,
        'sik': sik,
        'pay_pred': pay_pred
    })



if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)