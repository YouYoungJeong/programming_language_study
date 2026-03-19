"""
편차가 큰 데이터에 대한 로그 변환이 왜 필요한지 정리하기.
로지스틱 리그레션할때 로그 왜쓰는데?
-------
ML에서 데이터 분석시 log를 사용하면
1. scail(규모) 차이를 축소해 준다. log(10)=1,  log(100)=2, log(1000)=4
2. 로그변환하면 치우친 데이터를 정규분포에 가깝게 변경 가능.
3. 모델링에서 지수 관계를 선형관계로 바꿔준다. y = a*x^b => log(y) = log(a)+b(x)

정규화(nomalized : min-max-saciling)
(요소값 -최소값) / (최대값 - 최소값)
모든 피처(데이터)가 0~1사이에 위치하도록 하는 데이터

표준화(standardization)
(요소값 - 평균) / 표준편차
0주위에 표준편차 1의 값으로 배치되도록 피저 표준화함.
"""

import numpy as np
# 과학적 표기형식을 취소하기 = 3.45e+02
np.set_printoptions(suppress=True, precision=6) # 소수점 6자리 까지
def test():
    values = np.array([345, 34.5, 3.45, 0.345, 0.01, 10, 100])
    print(np.log2(3.45), ' ' , np.log10(3.45), " ", np.log(3.45))
    
    log_values = np.log10(values)   # 상용로그
    ln_values = np.log(values)      # 자연로그
    print("원본 값 : ",  values)
    print("log_values : ", log_values )
    print("ln_values : ", ln_values )


    # 정규화
    min_log = np.min(log_values)
    max_log = np.max(log_values)
    normalized = (log_values - min_log) / (max_log -min_log) 
    print("정규화 :" , normalized)


class LogTrans:
    # 편차가 큰 데이터 로그 스케일 변환하고  그 역변환을 제공하는 클래스
    # offset=1.0 -> 로그는 음수나 0이 나오면 X
    def __init__(self, offset:float=1.0):
        self.offset = offset
    
    # 로그변환 메소드 - 자연로그 스케일 줄이기
    def transform(self, x:np.ndarray)-> np.ndarray: # type hint:np.ndarry : 타입힌트를 줌 가독성up
        return np.log(x + self.offset)
    
    # 역변환 메서드 - 원복
    def inverse_trans(self, x_log:np.ndarray) -> np.ndarray:
        return np.exp(x_log)-self.offset


def main():
    test()
    print("***" * 10)
    data = np.array([0.001, 0.01, 0.1, 1, 10 , 100, 1000, 10000], dtype=float)

    log_trans = LogTrans(offset=1.0)
    data_log_scaled = log_trans.transform(data)                 # 로그 변환
    reversed_data = log_trans.inverse_trans(data_log_scaled) # 역변환
    print("원본 : ", data)
    print("로그 변환 : " , data_log_scaled)
    print("역변환 : ", reversed_data)


if __name__ == '__main__':
    main()