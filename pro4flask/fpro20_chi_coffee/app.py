from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path


# db.py 파일 모듈 가져오기
from db import insert_survey, fetchall_survey

# analysis.py 파일 모듈 가져오기
from analysis import analysiy_func, save_barchart_func

# 경로 설정하기
BASE_DIR = Path(__file__).resolve().parent
IMG_PATH = BASE_DIR/ 'static' / 'images' / 'vbar.png'

app = Flask(__name__)
@app.get("/")
def index():
    return render_template('index.html')



# 커피 선호도 조사
@app.get('/coffee/survey')
def survey_view():

    return render_template('coffee/coffeeservey.html')


# 설문조사값 받아 db저장후 출력
@app.post('/coffee/surveyprocess')
def surveyprocess():
    gender = (request.form.get("gender") or "").strip()
    age_raw = (request.form.get("age") or "").strip()
    co_survey = (request.form.get("co_survey") or "").strip()

    # 입력검증 초간단 흉내(흉내만~ 꼭하는게 좋음)
    if not gender or not co_survey or not age_raw.isdigit():
        return redirect(url_for("survey_view"))
    
    age = int(age_raw)

    # db에 자료 저장하기 - db.py모듈 insert_survey함수 사용
    insert_survey(gender=gender,age=age, co_survey=co_survey)

    # db 조회및 분석
    rdata = fetchall_survey()
    # print(rdata)
    crossTab, results, df = analysiy_func(rdata=rdata)

    # 차트 저장
    if not df.empty:
        save_barchart_func(df, IMG_PATH)


    # 보여주기
    return render_template("coffee/result.html",
                        crossTab = crossTab.to_html() if not crossTab.empty else '데이터가 없어요', 
                        results = results,
                        df = df.to_html(index=False) if not df.empty else '')
    

# 저장 없이 결과만 확인출력하는 라우트
@app.get("/coffee/surveyshow")
def survey_show():
    # db 조회및 분석
    rdata = fetchall_survey()
    crossTab, results, df = analysiy_func(rdata=rdata)

    # 차트 저장
    if not df.empty:
        save_barchart_func(df, IMG_PATH)
    
    # 보여주기
    return render_template("coffee/result.html",
                        crossTab = crossTab.to_html() if not crossTab.empty else '데이터가 없어요', 
                        results = results,
                        df = df.to_html(index=False) if not df.empty else '')



if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
