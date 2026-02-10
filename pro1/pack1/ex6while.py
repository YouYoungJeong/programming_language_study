# While : 조건에 의해 반복, 끝나는 횟수가 부정확할때 사용.


## while
a = 1           # 변수의 초기값
while a <= 5:   # 조건
    print(a, end='')    # -> 조건이 True여서 무한루팅(Looping, 반복)에 빠짐 ctrl+c
    a = a + 1   # ->  변수의 변화를 주어져야함
''' -> false조건을 만들어줘야 탈출,'''

## while else
while a <= 5:   
    print(a, end=', ')    
    a = a + 1
else:
    print("수행 성공")


## 다중 while
i = 1
while i <=3:
    j = 1
    print()
    while j <= 4:
        print('i='+ str(i) + '/j=' + str(j), end=(" , "))
        j = j + 1
    i = i + 1
print()
### 다중 while test
print('\n-----별찍기-----')
i = 1
while i <= 10:      # -> 1~10 행
    j = 1
    msg = ''
    while j <= i:   # -> 별의 갯수
        msg += '*'
        j += 1
    print(msg)
    i += 1

### test
print('--- 1~100 사이의 정수 중 3의 배수의 합 ---')
su = 1; hap = 0 # -> 변수의 위치 중요 hap이 while안으로 들어가면 리셋됨.
while su <= 100:   
    # print(su)
    if su % 3 == 0:
        # print(su)
        hap += su # -> 컴퓨터가 좋아하는 식 /사람이 좋아하는식 -> hap = hap + 1 <- 사람이 좋아하는식
    su += 1
print('합은 = ',hap)

### list 불러오기
colors = ['빨강' ,'파랑', '노랑']
'''
num = 0
print(colors[0])
print(colors[num]) # 인덱스 에리어
print(colors[1])
print(colors[2])
'''
num = 0
while num < 3:
    print(colors[num], end=' , ')
    num += 1

while num < len(colors):
    print(colors[num], end=' , ')
    num += 1

# 2교시 시작-------------------------------------------------------------------

### if문 안에 while : 블럭은 블럭을 포함O
print('if 블럭내 while블럭 사용---')
import time # -> 시간 재는 함수
sw = input('폭탄 스위치를 누를까요?[y/n]')
# print('sw : ', sw)
if sw == 'Y' or sw == 'y':
    # pass # -> 참일때 수행할 작없이 없을때 써줌 , 없으면 애러
    count = 5
    while 1 <= count:
        print('%d초 남았습니다.'%count)
        time.sleep(1) # -> 1초 후 다음 문장 실행.
        count -= 1
    print('폭발')
elif sw == "N" or sw == 'n':
    print('작업 취소')
else:
    print("y 또는 n을 누르세요.")

### while 기본끝


## continue 와 break
print()
print('\ncontinue /  break -----')
### 강제 종료 break
a = 0
while a < 10:
    a += 1
    
    if a == 3:          # -> a=3값은 안찍고 싶어 , 조건이 하나면 if a == 3: continue 한줄로 써도 됨.
        continue        # -> 아래 문장을 무시하고 자기와 대응되고 있는 while로 이동.
    if a == 5: continue

    if a == 7: break    # while문 무조건 탈출 : 강제종료되면 else를 안만남.
    print(a, end=(' , '))
else:
    print('정상 종료') # 조건에 의한 탈출
print('while 수행 후 a값 = %d'%a)

### 조건에 의한 탈출 else
a = 0
while a < 10:
    a += 1
    
    if a == 3:          # -> a=3값은 안찍고 싶어 , 조건이 하나면 if a == 3: continue 한줄로 써도 됨.
        continue        # -> 아래 문장을 무시하고 자기와 대응되고 있는 while로 이동.
    if a == 5: continue

    # if a == 7: break    # while문 무조건 탈출 : 강제종료되면 else를 안만남.
    print(a, end=(' , '))
else:
    print('정상 종료') # 조건에 의한 탈출
print('while 수행 후 a값 = %d'%a)


### 무한루트 while문 : html에서 많이 사용
print('\n 키보드로 숫자를 입력 받아홀수 짝수 확인하기(무한 반복)----')
while True:   #True(명시적) , 1, 100, -12, 4.5, 'ok' ..., 데이터 == 데이터가 있으면 True로 만들어서 무한루트로 만듦.  => 전체가 참인단어들이래.
    mysu = int(input('확인할 숫자 입력 (예:5, 0은 종료) : '))
    if mysu == 0:
        print('프로그램 종료')
        break
    elif mysu % 2 == 0 :
        print("%d는 짝수"%mysu)
        continue # 명시적으로 표현 의미는  X
    elif mysu % 2 == 1 :
        print("%d는 홀수"%mysu)




print("end")