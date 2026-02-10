# 기본 자료형 : int, float, bool, complex
# 묶음 자료형 : str, list, tuple, set, dict

# 1. str : 문자열 묶음 자료형. 순서O, 수정X
s = 'sequence'
print(s)
print(s, id(s)) #주소 -> sequence의 전체단어에 대한 주소가 아니라 제일앞's'의 주소만 가지고 있는것임
print('길이 : ', len(s))
print(s[0], s[2]) # sequence -> 0~7 : 주소는 's'만 가지고 있기 때문에 배열 순서로 찾아

print('길이 : ', s.find('e'), s.find('e',3), s.rfind('e')) # 함수 성격 검색해서 정리### 문자열 관련

##★인덱싱 / 슬라이싱
print(s[5]) # 인덱싱 : 변수명[순서]==인덱스 index는 0부터 출발
print(s[2:5]) # 슬라이싱 변수[s:e:s] {<-- 찾아보기}: 범위가 있음 s[이상 : 미만]
print(s[:], ' ', s[0:len(s)], s[::1])
print(s[0:7:2]) # 0부터 7미만까지 2씩 증가해
print(s[-1]) # -는 뒤에서부터셈. 끝단어가 -1
print(s[-1], " ", s[-4:-1:1])
print(s)
print(s,id(s)) #140711823505928
s = 'sequenc' # <- 이건 수정이X, 변경 -> string type는 수정이 안되고 주소가 바뀜
print(s)
print(s,id(s)) #2615745514784

print('--------')
# 2. List[] : 다양한 종류의 data 묶음 자료형. 순서O, 수정O, 중복O
a = [1, 2, 3]
print(a, a[0], a[0:3], a[0:2]) #순서가 있어서 소수점X, 인덱싱O, 슬라이싱O 
b = [10, a, 10, 20.5, True, '문자열'] # 중복O -> list[list] 가능
print(b)
print(b, ' ', b[1], " ", b[1][0]) #<- {다시 확인-> b id의 0번째는 주소}
print(id(b[0]))
print()
family = ['엄마', '아빠', '나', '여동생']
print(id(family))
print(family)

### list에 값추가
family.append('남동생')
print(id(family))
print(family)
family.remove('나') # 삭제
print(family)
family.insert(0,'할머니') #삽입
print(family)
family.extend(['삼촌','고모', '조카']) #추가
print(family)
family += ['이모'] # 추가
print(family)
print(family.index('아빠'))
print('엄마' in family, '나' in family) #{in 확인}
family.remove('아빠')  # 값에 의한 삭제
del family[2] # 순서에 의한 삭제
print(family)

### sort 정렬 : 원본이 바뀜
print()
kbs = ['123', '34', '234']
kbs.sort() # 문자열 정렬 (ascending sort : 오름차순)
print(kbs)
mbc = [123, 34, 234]
mbc.sort() # 숫자열 정렬 (dscending sort : 내림차순)
print(mbc)
### sorted 정렬 : 원본 유지, sorted된 자료를 원본 유지된체 새로운 객체에 들어감.{단어 정리}
sbs = [123, 34, 234]
ytn = sorted(sbs)
print(sbs)
print(ytn) #[34, 123, 234]

print()
### 치환, copy
name = ['tom', 'james', 'oscar']
name2 = name
print(name, id(name))
print(name2, id(name2)) # 치환은 주소가 같음
import copy # copy.deepcopy모듈을 사용하면 새로운 객체 -> 주소가 달라짐
name3 = copy.deepcopy(name)
print(name3, id(name3))
name[0] = '길동'
print(name)
print(name2)
print(name3) # ['tom', 'james', 'oscar'], 새로운 객체여서 바뀌지X


# 3. tuple() : list와 유사, 읽기 전용.  순서O, 수정X, 중복O (Read only, 검색속도 빠름)
t = (1,2,3,4) # ,<- 괄호 쓰는것을 권장
t = 1,2,3,4 # 위와 동일
print(t, type(t)) #(1, 2, 3, 4) <class 'tuple'>
k = (1)
print(k, type(k)) #1 <class 'int'> -> 하나만 있을때는 tuple X
k = (1, )
print(k, type(k)) #(1,) <class 'tuple'>
print(t[0], ' ', t[0:2]) # 인덱싱, 슬라이싱 O -> tuple이구나!
# t[0] = 77 #TypeError: 'tuple' object does not support item assignment:수정 불가
### 자료형의 변형
imsi = list(t) #list로 변형
imsi[0] =77 # 값 변경
t = tuple(imsi) # tuple로 변경
print(t)

print("---"*10)

# 4. set : 순서X, 중복X, 수정O
ss = {1, 2, 1, 3}
print(ss)
ss2 = {3, 4}
print(ss.union(ss2)) # 합집합
print(ss.intersection(ss2)) # 교집합
print(ss - ss2, ss | ss2, ss & ss2) # 차, 합 교
# print(ss[0]) # 순서 X
ss.update({6, 7})
print(ss)
ss.discard(7) # 값 삭제
ss.discard(7)# 값 삭제 discard는 있으면 지우고 없으면 pass
ss.remove(6)# 값 삭제
# ss.remove(6) # remove는 있으면 지우고 해당 값이 없으면 에러
print(ss)

li = ['aa', 'aa', 'bb', 'cc', 'aa']
print(li)
imsi = set(li) # set으로 변환해서 중복 제거, 순서는 지맘대로
li=list(imsi)
print(li)


print("---"*10)
# 5. dict : 사전 자료형 {'키':값} 형태 ,수정O,순서X(업데이트로 인해 입력순서대로 출력되지만 순서는 없다.), 키를 가지고 값을 찾아내는것, 
## 방법1
mydic = dict(k1 = 1, k2 ='ok', k3=123.4)
print(mydic, type(mydic)) # {'k1': 1, 'k2': 'ok', 'k3': 123.4} , <class 'dict'>
##방법2 좀더 명시적
dic = {'파이썬':'뱀', '자바':'커피', '인사':'안녕'}
print(dic)
print(len(dic))
print(dic['자바']) #키로 값을 겁색
ff=dic.get('자바') # get으로 겁색
print(ff)
# print(dic['커피']) # 키가 아니라 값이라 Error

# print(dic[0]) # 순서X -> 인덱싱이 없어
dic['금요일'] = '와우'
print(dic) # {'파이썬': '뱀', '자바': '커피', '인사': '안녕', '금요일': '와우'} 추가라는 개념보다는 삽입.
del dic['인사'] # 삭제
print(dic)
print(dic.keys())
print(dic.values())