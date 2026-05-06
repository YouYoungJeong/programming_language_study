# model.py
import numpy as np
from tensorflow.keras.models import load_model
import os

# 저장된 모델 경로
MODEL_PATH = '../classif_quiz/seq_model.keras'

# Flask 실행 시 모델 1번만 로드
model = load_model(MODEL_PATH)


def predict_diabetes(data):
    """
    main.html에서 Ajax로 받은 입력값을 이용해
    당뇨병 발생 여부를 예측하는 함수

    Parameters
    ----------
    data : dict
        사용자가 입력한 8개 feature 값

    Returns
    -------
    result : dict
        예측 확률, 예측 클래스, 판정 결과
    """

    # 입력값 순서는 학습 데이터의 컬럼 순서와 반드시 같아야 함
    input_data = np.array([[
        float(data["pregnancies"]),
        float(data["glucose"]),
        float(data["blood_pressure"]),
        float(data["skin_thickness"]),
        float(data["insulin"]),
        float(data["bmi"]),
        float(data["diabetes_pedigree"]),
        float(data["age"])
    ]])

    # sigmoid 출력값: 0~1 사이 확률
    pred_prob = model.predict(input_data, verbose=0)[0][0]

    # 0.5 이상이면 당뇨 가능성 있음
    pred_class = 1 if pred_prob >= 0.5 else 0

    if pred_class == 1:
        message = "당뇨병 발생 가능성이 있습니다."
    else:
        message = "당뇨병 발생 가능성이 낮습니다."

    return {
        "probability": round(float(pred_prob), 4),
        "prediction": pred_class,
        "message": message
    }