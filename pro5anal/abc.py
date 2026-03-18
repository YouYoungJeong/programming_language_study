# 워밍업!
'''
통계량 : 데이터의 특징을 하나의 숫자로 요약한 것.
표본 데이터를 추출해 전체(모집단) 데이터를 짐작 가능
평균, 분산, 표준편차 ...
통계는 분산의 마법이다 통계는 분산 없이 결과가 나오지 X - 분산이 나오려면 평균 필수지.

편차 제곱의 평균
원래의 편차값에 root 표준편차
'''

grades =  [1, 3, -2, 4]         # 변량 숫자로 표현할 수 있는 자료
def show_grades(grades):
    for g in grades:
        print(g, end=" ")

show_grades(grades)

def grades_sum(grades):
    tot = 0
    for g in grades:
        tot += g
    return tot
print(f"\n\n합은 = {grades_sum(grades)}")

def grades_ave(grades):
    ave = grades_sum(grades) / len(grades)
    return ave
print(f"평균은 = {grades_ave(grades)}")

# 분산 (편차 제곱의 평균) : 평균값 기준으로 다른 값 들의 흩어진 정도를 나타냄.
def grades_variance(grades):
    ave = grades_ave(grades)
    vari = 0
    for su in grades: 
        vari += (su - ave) ** 2     # 편차 구하기
    return vari/len(grades)         # 표본의 갯수 전체 갯수로 나눔
    # return vari/(len(grades)-1)   # 표본의 갯수 전체 갯수 -1 로 나눔
                                    # 빅데이터로 사용할때 데이터의 량이 많이 때문에 
                                    # 전체갯수(python), 전체갯수 -1(R) 로 나눠도 차이가 많이 없다 
                                    # 그래도 차이가 남

print(f"분산은 = {grades_variance(grades)}")

# 표준편차
def grades_std(grades):
    return grades_variance(grades) ** 0.5 # ** 0.5 = root
print(f"표준편차는 = {grades_std(grades)}")


# numpy 함수 사용하기
print("\nnumpy 사용")
import numpy
print(f"합은        = {numpy.sum(grades)}")
print(f"평균 은     = {numpy.mean(grades)}")
print(f"분산 은     = {numpy.var(grades)}")
print(f"표준 편차는 = {numpy.std(grades)}")