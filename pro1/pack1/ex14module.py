'''
Module :소스코드의 재사용을 가능하게 하며, 소스코드를 하나의 이름 공간으로 구분하고 관리
        이미 있는 모듈을 잘 사용 ,수정 할 수 있기위해 공부하는것,
***.py 는 하나의 파일은 하나의 모듈이 됨
모듈의 종류 : 
- 표준 모듈(내장 모듈-python설치하면 기본적으로 깔려있음(내장 함수도)
- 사용자 작성 모듈,
- 제3자 모듈(third party):(site-packages(보조 기억장치)에 들어있는 모듈)

모듈의 의 맴버 : 변수는 전역변수, 명령문 제어문, 함수, 클래스

모듈의 방법 :  독립적 시행, 호출
__main__ : 모듈이 여러개 일때 모듈의 시작점 Main모듈  = 응용 프로그램의 시작점 
***.pyd, ****.dll 은 C Library 파일 python에서 사용

모듈 생성시 편집기에 따라 약간 달라짐(VSCode, Pycham...)
'''



print(print.__module__) # print는  builtins 이 모듈안에 들어있음

# 외부 모듈 사용하기
import sys #import 모듈명

print(sys.path) # 모듈명.맴버 (): 없으면 sys.의 맴버구나

## 강제종료
'''
def의 return은 무조건 빠져나와
sys.exit는 강제종료 밑으로 안내려가
'''
a =  5
if a > 7:
    sys.exit() # 응용 프로그램 강제종료. ():함수 아니면 매서드구나 까봐야 알아
import math
print(math.pi)

### improt , del
import calendar
calendar.setfirstweekday(6)
calendar.prmonth(2026,2)
del calendar # import -> dle 메모리에서 떨어져 나감.

### from, import , * 사용하기
import random 
print(random.random()) # 0~1 사이의 실수 
print(random.randrange(1,10)) # 1 ~10까지의 정수


from random import random, choice, randrange
# from random import * : 램을 많이 사용함. 비추
print(random())


print('end') # 응용프로그램 종료 : 램이 깨끗해짐.
