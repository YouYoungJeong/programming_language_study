'''
문1 ) 처리 함수 : processfunc(datas) : datas에 기억된 내용을 출력한다.
처리 조건 : 

 1) 급여액 =  기본급 + 근속수당 

 2) 수령액 =  급여액 - 공제액

* 근무년수에 대한 수당표	
근무년수     근속수당
0~3년        150000
4~8년        450000
9년 이상    1000000	

 * 급여 상한액에 대한 
공제세율표급여액                공제세율
300만원 이상          0.5
200만원 이상          0.3
200만원 미만          0.15
출력 결과 : 

사번  이름    기본급    근무년수  근속수당  공제액    수령액
-------------------------------------------------------------------------------
1    강나루    1500000   16       1000000   750000   1750000
2    이바다    2200000   8        450000    795000   1855000
3    박하늘    3200000   21       1000000   2100000  2100000
처리 건수 : 4 건
'''
# 입력 함수 :  [사번, 이름, 기본급, 입사년도]
def inputfunc():
    datas = [
        [1, "강나루", 1500000, 2010],
        [2, "이바다", 2200000, 2018],
        [3, "박하늘", 3200000, 2005],
    ]
    return datas

# print(f'사번: {} , 이름 :{} ,  기본급:{}, 근무년수 : {} , 근속수당 : {} , 공제액 : {},  수령액 : {}')
def processfunc2(datas):

    # 개인 리스트 추출후 근속기간 정리
    for sebu in datas: 
        sebu.append(2026-sebu[3])

        # 근속수당, 급여액 계산 리스트 추가
        if sebu[4] <= 3:
            sebu.append(150000)
            sebu.append(150000+sebu[2])
        elif 3 < sebu[4] <= 8:
            sebu.append(450000)
            sebu.append(450000+sebu[2])
        else:
            sebu.append(1000000)
            sebu.append(1000000+sebu[2])

        # 공제액 계산
        if sebu[6] <= 2000000:
            sebu.append(sebu[6]*0.15)
        elif 2000000 < sebu[6] < 3000000:
            sebu.append(sebu[6]*0.3)
        else:
            sebu.append(sebu[6]*0.5)
        sebu.append(sebu[6]-sebu[7])
    

        # print(sebu[6]-sebu[7]) 
        # print(2026-sebu[3])
        result = print(f'사번: {sebu[0]} , 이름 :{sebu[1]} ,  기본급:{sebu[2]}, 근무년수 : {sebu[4]} , 근속수당 : {sebu[5]} , 공제액 : {sebu[7]} ,수령액 : {sebu[8]}')
    print('처리건수 : ',len(datas))
    return result   
    
print(processfunc2(inputfunc()))

# 풀이 -------------------------------------------------------------------------------------------
print('----------'*10)
def inputfunc():
    datas = [
        [1, "강나루", 1500000, 2010],
        [2, "이바다", 2200000, 2018],
        [3, "박하늘", 3200000, 2005],
    ]
    return datas
# 함수를 너무 길게 만들지말고 unit단위로 만들어서 붙여서 써.
# 자원의 재활용이 이런 의미.
datas= inputfunc()
import time #을 쓰는걸 추천
year = time.localtime()[0]
# def processfunc(datas):
    
    # CURRENT_YEAR = datetime.now().year

# print(processfunc(datas)
# 
print(year)