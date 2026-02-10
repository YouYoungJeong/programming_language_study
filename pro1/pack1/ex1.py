# 가상환경 연결, 출력 하는 법
var1 = "안녕 파이썬"
""" 
뭘쓸지정해야함 -->명령팔레트 >select interpreter > myproject 사용 > 
새터미널 선택> 파일위치로 이동후 >python 파일명.py명령어 입력//python 파일명.py < 여기에 python은표준 출력장치,콘솔로 출력하기 위한 장치
"""
print(var1) 
## (주석) ctrl + /
"""
여러줄 주석
"""
"""
python은 ",' 구분이 없어.
"""

var1 = 5; var2 = 6 # 한줄에 변수 두개 주는건 안좋아. ;으로 구분가능

var1 = 5
var2 = 6; # 문장 끝에; 쓰는건 옵션
"""
참조용 기억장소 : var1 = 5 -> 5는 어떤 기억장소에 있고 var1(객체:object)은 5에대한 주소를 가지고 있음 
그래서 var1은 아무타입이나 가능하다 처음부터 int, str같은거를 설정 할 필요가 없다.
Python은 모두다 참조용 기억장소, 기본형이라는 단어가 아예 없다.
기본형 기억장소 : int 변수 = 5, str 변수 = abc 
"""
var1 = 10
print(var1) # ctrl + `로 터미널 왔다갔다 하기
var1 = 5.6
print(var1)
var2 = var1
print(var1, var2) # var1, var2는 같은 주소를 가지고 있고 동일한 객체를 가지고 있다.
var3 = 7
print(var1, var2, var3)

##id()
print(id(var1), id(var2), id(var3)) # id 참고하고있는 객체의 주소

##변수명 설정할때
Var3 = 8
print(var3, Var3) # 변수명(함수,클래스)은 대소문자 구분을 한다, 특수문자'_'를 제외하고 사용X,숫자로 시작X

## is : 주소 비교 연산자 , == : 값 비교연산(치환)
a = 5
b = a
c = 5
print(a, b, c)
print(a is b, a == b)
print(b is c, b == c)

##[] = list : 여러가지의 요소값을 갖는다 -> 집합형 요소
aa = [5]
bb = [5]
print(aa, bb)
print(aa is bb, aa == bb) #값은 같지만 주소를 달라

print('--------') # == print("--------")
import keyword # 키워드 목록 확인용 모듈 읽기
print('예약어 목록:', keyword.kwlist) 
""" 
['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 
'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 
'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 
'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'
다 기능이 있는 단어기 때문에 변수명명 으로 쓰지 X
"""
# for = 4  SyntaxError: invalid syntax

##type 자료형 확인, isinstance
print('type(자료형) 확인')
kbs = 9
print(isinstance(kbs, int)) # True
print(isinstance(kbs, float)) # False
##--- 기본 데이터 ---
print(5, type(5))               # 5 <class 'int'> ; class = 객체(object)라는 뜻
print(5.2, type(5.2))           # 5.2 <class 'float'>
print(3 + 4j, type(3 + 4j))     #(3+4j) <class 'complex'>
print(True, type(True))         #True <class 'bool'>
print('good', type('good'))     #good <class 'str'>
##--- 묶음형 data ---
print((1,), type((1,)))           #(1,) <class 'tuple'>
print([1], type([1]))           #[1] <class 'list'>
print({1}, type({1}))           #{1} <class 'set'>
print({'k':1}, type({'k':1}))   #{'k': 1} <class 'dict'>