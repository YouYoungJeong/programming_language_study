# 반복문  for : 묶음형 자료가 주로 씀


## 기본 for문 구조 
for i in [1, 2, 3, 4 ,5] : # -> list가 object(객체) , 묶음형 자료를 하나씩 꺼내서 사용.
    print(i, end="  ")

print()
for i in (1, 2, 3, 4 ,5) : # tuple
    print(i, end=" ")

print()
for i in {1, 2, 3, 4 ,5} : # set : 순서대로 안찍을 수 있다.
    print(i, end="  ")

## 분산 표준편차 구경하기
print('\n분산 / 표준편차 --------') # numpy에서 제대로 배움.
### 합/ 평균 구하기
# numbers = [1, 3, 5, 7, 9]     # 합은25, 평균은5.0
# numbers =[3,4,5,6,7]          # 합은25, 평균은5.0
numbers = [-3, 4, 5, 7, 12]     # 합은25, 평균은5.0
tot = 0
for a in numbers:
    tot += a
print(f"합은{tot}, 평균은{tot / len(numbers)}")
avg = tot / len(numbers) # 평균값
### 편차의 합
hap = 0
for i in numbers:
    hap += (i - avg) ** 2 #편차 제곱의 합
print(f'편차 제곱의 합 {hap}')
###분산 구하기
vari = hap / len(numbers)
print(f'분산 {vari}')
###표준편차 구하기
print(f'표준편차 {vari ** 0.5}')


print()
colors = ['r', 'g','b']
for v in colors :
    print(v, end=("  "))


## ** iter() , enumerate() 
### iter()
print('iter() : 반복 가능한 객체를 하나씩 꺼낼 수 있는 상태로 만들어 주는 함수')
iterator = iter(colors)
for v in iterator:
    print(v, end=" , ")
### enumerate()
print()
for idx , d in enumerate(colors): # index와 값을 반환
    print(idx, ' ', d)

### comprehension
print()
print('comprehension : 반복문 + 조건문 + 값 생성을 한 줄로 표현')
a = [1,2,3,4,5,6,7,8,9,10]
li = []
for i in a:
    if i % 2 == 0:
        li.append(i)
print(li)
''' 둘이 같은 문장 '''
print(list(i for i in a if i % 2 == 0)) # => 구조 파악 연습!

#### comprehension 연습1
datas = [1, 2, 'a', True, 3.0]
li2 = [i * i for i in datas if type(i) == int] # -> python은 이런 문장 多
print(li2) #[1, 4]

datas = {1, 2, 'a', True, 3.0, 2, 1, 2, 1, 2, 2}
li2 = [i * i for i in datas if type(i) == int]
print(li2) #[1, 4] -> set중복 배제 unique)

id_name = {1:'tom', 2:"oscar"}
print(id_name)
name_id={val:key for key, val in id_name.items()} #-> dict key:value 바꾸기
print(name_id)

### comprehension과 unpack
print()
print([1,2,3])  # [1, 2, 3]
print(*[1,2,3]) # *: unpack ->  1 2 3
aa = [(1,2), (3,4), (5,6)]
for a, b in aa:
    print(a + b,) # 3 7 11

print()
print([a + b for a, b in aa], sep='\n') #[3, 7, 11]
print(*[a + b for a, b in aa], sep='\n') # 3 7 11




##dict타입 for문
print()
print('사전형(dict) --------')
datas = {'python':'만능언어', 'java':'웹용언어', 'mariadb':'RDBMS'}
for i in datas.items():
    # print(i, end=" ") # tuple으로 반환
    print(i[0], '~~', i[1], end=" ")

print()
for k ,v in datas.items():
    print(k, '~~', v, end=" ")

print()
for k in datas.keys():
    print(k, end=" ")

print()
for v in datas.values():
    print(v, end=" ")


## 다중 for문
print()
print('다중 for문')
for n in[2, 3]:
    print('--{}단'.format(n))
    for i in [1, 2 ,3 ,4 , 5, 6, 7, 8, 9]:
        print('{} * {} = {}'.format(n, i, n*i))

## for문 continue, break
print('continue, break')
nums = [1,2,3,4,5]
for i in nums:
    if i == 2 : continue
    if i == 4 : break
    print(i, end=" ")
else:
    print('정상 종료')


## 정규표현식 연습 + for
print()
print('정규표현식 연습 + for')
str = """밀가루와 설탕, 전기 등 민생과 직결된 생필품 시장에서 수년간 짬짜미를 벌여 물가 상승을 초래한 혐의로 업체 관계자들이 무더기로 재판에 넘겨졌습니다.서울중앙지검 공정거래조사부는 지난해 9월부터 생필품 담합 사건을 집중 수사한 결과, 시장 질서를 교란하고 서민 경제를 위협한 혐의로 업체 관계자 등 총 52명을 기소했다고 오늘(2일) 밝혔습니다."""

import re # 정규표현식은 r''선행
str2 = re.sub(r'[^가-힣\s]',"", str) # 한글과 공백 이외의 문자는 공백처리
print(str2)
str3 = str2.split(' ') # 공백을 기준으로 자르고 list에 들어감.
print(str3)
cou = {} # set or dict :들어오는 값에 따라 타입 결정
### 중복확인용 dict만들기
for i in str3:
    if i in cou:
        cou[i] += 1 # 같은 단어가 있으면 누적
    else:
        cou[i] = 1 # 최초 단어인 경우 '단어: 1'
print(cou)

print()
print('정규표현식 좀 더 연습')
for test_ss in ['111-1234', '일이삼-일이삼사','222-1234', '333&1234']:
    if re.match(r'^\d{3}-\d{4}$', test_ss): # ^앞에 있으면 처음에 라는 뜻, \d : 숫자 ,{}: 갯수,  $는 마지막 => 숫자로 시작하고 숫자로 끝나.
        print(test_ss, '전화번호 맞아요')
    else:
        print(test_ss, '전화번호 아니야')

## 수열 생성 range
print()
print('수열 생성 : range')
print(list(range(1,6)))     # [1, 2, 3, 4, 5] 1이상 6미만, 증가치1은 생략가능=(1,6,1), 반환타입 설정해줘야함<- python3.~ 부터 바뀜.
print(tuple(range(1,6,2)))  # (1, 3, 5)
print(tuple(range(1,6,2)))
print(list(range(-10,-100, -20))) # [-10, -30, -50, -70, -90]
print(set(range(1,6)))

for i in range(6) :     # 시작값을 안주면 0 부터 시작, 증가치 안주면 1
    print(i, end=" ")   # 0 1 2 3 4 5
print()
for _ in range(6) :     # _ : 변수를 사용 하지 X 반복만 할때 사용
    print("반복")

tot = 0
for i in range(1,11):
    tot += i
print('tot = ', tot)
print('tot = ', sum(range(1,11))) #sum 내장 함수 사용

for i in range(1, 10):
    print(f'2* {i} = {2 * i}')


'''for문 사용'''
# 문제1 : 2~9 구구단 출력 (단은 행 단위 출력)


gugu=1
for i in range(2,10):
    gugu += 1
    print(gugu,'단')
    for num in range(1,10):
        print('{} * {} = {}'.format(gugu,num,gugu * num),end=' , ')
    print('\n')


# 문제2 : 주사위를 두번 던져서 나온 숫자들의 합이 4의 배수가 되는 경우만 출력

dicesum = {}
for dice1 in range(1,7):
    for dice2 in range(1,7): 
        # print(dice1, dice2, dice1+ dice2)
        if (dice1+dice2) % 4 ==0:
            print(dice1,"+",dice2,"=",(dice1+dice2))


for i in range(1,7,1):
    for j in range(1,7,1):
        su = i+j
        if su % 4 ==0:print(i,j)
print('\nend')


# 수치연산은 C/Python이 좋다, java는 웹전용.