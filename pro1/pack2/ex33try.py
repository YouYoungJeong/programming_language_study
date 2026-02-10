'''
예외처리 : 파일, 네트워크, DB작업, 실행에러 등의 에러 대처
파일, 네트워크, DB작업 : 외부 장치와 연결하는 작업 - 예외처리 필수!
    - try , except, finally
'''

def divide(a , b):
    return a / b

print('이런 저런 작업 진행...')
# c = divide(5 , 2)
# c = divide(5 , 0)
# print(c)


# Exception은 super
    # Name, Index, Value, FileNotFound, ZeroDivision Error클래스들의 

# try 문은 쓸까 말까 고민하면 써라(권장)
try:
    # ZeroDivisionError
    c = divide(5 , 2)                         # 실행문 (예외 발생 가능 구문, 값은 고정값이 아님.)
    # c = divide(5 , 0)                           
    print(c)

    # IndexError
    aa = [1, 2]
    print(aa[0])
    # print(aa[3])

    #FileNotFoundError
    open('c:/work/abc.txt')

# 오류 처리 클래스 순서 중요!

# 특정 오류 처리
except ZeroDivisionError:                       # <- 예외 종류 관련 클래스 적고
    print('두번째 값은 0을 주면 안돼요.')         # < -예외 처리
except IndexError as err:                       # 오류의 정확한 정보를 받으려면 as 로 받기
    print('참조 범위 오류 : ', err)


# 전체 에러 처리
# 에러 마다 일일이 쓰기 어려워, 최상위 클래스인 Exception을 사용
except Exception as e:
    print('Error : ', e)

finally:
    print('에러 유무에 상관없이 반드시 수행')



print('end')
print('종료')