from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
"""
그래프 저장 경로 설정 모듈 사용하기
matplotlib.use('Agg')
    Anti Grain Geometry : matplotlib의 랜더링 엔진중 하나
    이미지 저장시 오류 방지 - 차트 출력 없이 저장할 때 사용
"""
app = Flask(__name__)
# =============== 경로 상수 지정하기 ===================
# 현재 파일이 있는 경로 지정하기.
BASE_DIR = Path(__file__) .resolve().parent
# static/image경로 지정
STATIC_DIR = BASE_DIR / 'static' / 'images'
# 파일을 만드는데 있으면 에러나지말고 없으면 만들어
STATIC_DIR.mkdir(parents=True, exist_ok=True)
# ==========================================

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/showdata')
def showdata():
    df = sns.load_dataset('iris')
    print(df.head())

    # pie chart 생성 및 저장, 서버에서 자체 출력 X
    counts = df['species'].value_counts().sort_index()
    plt.figure()
    counts.plot.pie(autopct='%1.1f%%', startangle=90) # pie시작지점은 3시인데 지금 12시로 맞춤
    plt.tight_layout()
    img_path = STATIC_DIR / 'fpro19.png'
    plt.savefig(img_path, dpi=130) # dpi 이미지 해상도
    plt.close()

    irishtml = df.to_html(
        classes='table table-striped table-sm' ,# 스타일(css)
        index=False
    )
    return render_template('show.html',
                            table=irishtml,
                            img_path='images/fpro19.png' # 이미지의 경로를 넘길 수 있다.
                            )


if __name__ == '__main__':
    app.run(debug=True)