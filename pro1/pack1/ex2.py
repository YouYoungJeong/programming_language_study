# 연산자
v1 = 3  # 치환연산자
v1 = v2 = v3 = 5
print(v1, v2, v3)

##print
print('출력1')
print('출력2') # print 출력 하고 Enter
## end 출력
print('출력1', end=',')
print('출력2')
print('출력3')

v1, v2 = 10, 20 # 변수명만큼만 데이터 설정.
print(v1, v2)
v2, v1 = v1, v2 # 기본형은 불가능함. 참조형이라서 가능한 변수 지정.
print(v1, v2)

## packing 연산
print('값 할당 packing  연산')
v1 = 1,2,3,4,5  # == v1(1,2,3,4,5) 그룹할당
v1 = [1,2,3,4,5]
*v1, v2 = [1,2,3,4,5] 
print(v1, ' ', v2) # dir aa.py / dir aa.*(모든파일)
# v1, v2* = [1,2,3,4,5] # SyntaxError: invalid syntax
print(v1, ' ', v2)
*v1, v2, v3 = [1,2,3,4,5] 
print(v1, ' ', v2, ' ', v3)


##print(format()) - 정리 필요
'''https://cafe.daum.net/flowlife/RUrO/50'''
print()
print(format(1.5678, '10.3f')) # 전체 10자리의 데이터값에서 소수 이하 3째자리까지 찍어
print('나는 나이가 %d 이다.'%23) # %d 는 숫자열의 데이터값
print('나는 나이가 %s 이다.'%'스물셋') #%s 글자형
print('나는 나이가 %d 이고 이름은 %s이다.'%(23, '홍길동'))
print('나는 나이가 %s 이고 이름은 %s이다.'%(23, '홍길동'))
print('나는 키가 %f이고, 에너지가 %d%%.'%(177.7, 100)) #%f 는 실수형
print('이름은 {0}, 나이는 {1}'.format('한국인', 33))
print('이름은 {}, 나이는 {}'.format('신선해', 33))
print('이름은 {1}, 나이는 {0}'.format(34, '강나루'))
##print(f"") - 이방법을 더 많이 씀.
abc = 123
print(f"abc의 값은 {abc}임")
#이스케이프 문자\n, \b, \t
print('\n\n본격적 연산-------------') #\n : line skip, \b : back, \t : tap ...

##연산
### 산술 연산자
print(5 + 3, 5 - 3, 5 * 3, 5 / 3, 5 // 3, 5 % 3, 5 ** 3)
'''8 2 15 1.6666666666666667:/ 실수형 나누기 1://몫만 2:%나머지 나누기 125'''
print(divmod(5 , 3), ' ', 5 % 3) #(1, 2): 몫과 나머지   2 : 나머지
result = 3 + 4 * 5 + (2 + 3) / 2
print(result) 
# 연산자 우선순위 
#() -> **-> 단항->산술연산:[ * , /:왼쪽부터 수행 > +,- : 왼쪽부터 수행 ] ->관계연산 ->논리 연산(not > and > or)-> =(치환이 마지막)
### 비교연산자
print(5 > 3, 5 == 3, 5 != 3) # True False True
###논리 연산자
print(5 > 3 and 4 < 3 , 5 > 3 or 4 < 3, not(5 >= 3)) # False True False
print(True or False and False) 
print(True and False or False) #and 가 or 보다 우선순위 높다.
print(True and False)
### 문자열 연산
print(4 + 5) # 산술연산
print('4' + "5") # 문자열 더하기 연산  <- 문자열
print('한' + '국' + '만세')
print("한국" * 5) # 문자열은 더하기 곱하기(몇번을 더해라) 가능

## 누적
print('누적')
a , b, c, d= 10, 10, 10, 10
a = a + 1
a += 1  # -=, *=, /=
b -= 1
c *= 1
d /= 1
print(f"a는 {a}")
print(f"b는 {b}")
print(f"c는 {c}")
print(f"d는 {d}")
# print(a--) # ++, -- : Python은 증감 연산자 X

##부호 연산자 : 변수에 - 달리면 부호가 바뀜 - 설명 찾기
print(--a)
print(-a)
print(a * -1)

##
print(1 + 1)
# print(('1'+'1') + 1) # TypeError: can only concatenate str (not "int") to str
print(int('1'+'1') + 1) # 12 -> 문자열을 int로 바꾸면 정수 처리됨
# print((1 + 1) + '1') # TypeError: unsupported operand type(s) for +: 'int' and 'str'
print(str(1 + 1) + '1') # 21 str(숫자) -> 문자화
print(float('1'+'1') + 1) # 12.0 -> 문자열을 float로 바꾸면 실수 처리됨

##bool
print('boolean 처리 :', bool(True), bool(False))
print(bool(1), bool(12.3), bool('ok'), bool([12])) # True True True True
print(bool(0), bool(0.0), bool(''), bool([]), bool(None)) # False False False False False -> 데이터의 유무?

##★ r 선행문자 : 이스케이프 문자를 순수 data로 인식이 됨.
print('aa\tbb')
print('aa\nbb')
print(r'aa\tbb')
print(r'aa\nbb')