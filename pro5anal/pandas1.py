"""
고수준의 자료구조(Series, DataFrame)와 빠르고 쉬운 데이터 분석용 함수 제공
통합된 시계열 연산, 축약연산, 누락 데이터 처리, SQL, 시각화... 등을 제공
numpy 기반 - 연산관련된 모든 내장함수는 numpy를 사용
데이터 먼징(Data Munging) 또는 데이터 랭글링(Data Wrangling)을 효율적으로 처리

데이터 먼징(Data Munging) 또는 데이터 랭글링(Data Wrangling)
분석, 모델링, 시각화 등 후속 작업을 위해 
원시 데이터(Raw Data)를 정리, 구조화, 변환하여 사용 가능한 형태로 만드는 
필수 전처리 과정. 
불일치 처리, 데이터 타입 변환, 결측치 처리, 데이터셋 병합 등을 포함하여 
데이터의 품질과 가치를 향상
Series가 모여야 dataframe가 된다.
** 시리즈와 데이터프레임과의 상관관계 중요 **

numpy는 인덱스가 묵시적인데
pandas는 명시적 - Series
"""
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# Series : 일련의 객체를 담을 수 있는 1차원 배열과 같은 자료구조로 색인(index)을 갖음.
# 리스트를 가지고 Series 만듦 - 일반적으로 가장 많이 사용
obj = pd.Series([3, 7, -5, 4]) # (요소)dtype: int64
# 튜플을 가지고 Series 만듦
obj = pd.Series((3, 7, -5, 4))
# set을 가지고 Series 만듦 ->  ERR 순서가 X - TypeError
print(obj)
obj = pd.Series([3, 7, -5, "4"]) # (요소)dtype: object => 뭐든지 들어갈 수 있다는 뜻
print(obj, type(obj)) # dtype: object <class 'pandas.core.series.Series'>

obj2 = pd.Series([3, 7, -5, 4], index=["a","b","c","d"])
print(obj2)
print(obj2.sum(), " ", np.sum(obj2) ,' ', sum(obj2)) # pandas numpy python의 sum이지만 
                                                     # pandas내부에서 numpy의 sum을 쓰고 있는것 
print(obj2.std())

# 배열에 가지고 있는 값을 가져오기
print(obj2.values)      #[ 3  7 -5  4]
# 인덱싱 출력
print(obj2.index)       # Index(['a', 'b', 'c', 'd'], dtype='object')
# 슬라이싱 하기
print(obj2['a'])        # 3
# [['a']] :  'a'인덱스 값 출력
print(obj2[['a']])      # a    3

print(obj2[['a','b']])  # b    7
print(obj2['a':'c'])    # c   -5

# 인덱스 사용도 가능하다
print(obj2[2])          # -5
print(obj2[1:4])

# iloc를 사용해서 인덱스 값을 얻기 
print(obj2.iloc[2])     # -5
print(obj2[[2,1]])
print(obj2.iloc[[2,1]])

# 인덱스의 유무 판단
print('a' in  obj2)
# 인덱스의 유무 판단
print('k' in  obj2)

# 딕트 자료형 사용
print('파이썬 dict 자료를 Series객체로 생성')
names = {'mouse':5000, 'keyboard':25000, 'monitor':450000}
print(names)
# Series dict 생성 -> mouse(key:index)       5000(value:value)
obj3 = Series(names)
print(obj3)
# 인덱스값 바꾸기
obj3.index=['마우스', '키보드', '모니터']
print(obj3)

# Series객체에 이름을 생성
obj3.name = "상품가격"
print(obj3)

print('\n--------------DataFrame 객체-----------')
df = pd.DataFrame(obj3)
print(df, " ", type(df)) # <class 'pandas.core.frame.DataFrame'>
# dict로 df만들기 안에 값은 전부 vacter
data = {
    "irum" : ['홍길동','한국인','신기해','공기밥','한가해'],
    'juso' : ('역삼동','신당동','역삼동','역삼동','신사동'), # 튜플 가능
    "nai" : [23, 25, 44, 31, 35],
}
frame = pd.DataFrame(data)
print(frame)
print()
# 열 호출 방법
print(frame['irum']) # 가독성이 좋아서 많이 사용함.
print(frame.irum) # 객체의 멤버로 보이기 때문에 가독성이 떨어짐
# <class 'pandas.core.series.Series'> 세개의 시리즈가 만나 df를 생성.
print(type(frame.irum))
# 칼럼 순서 바꿔서 출력
print(DataFrame(data=data, columns=['juso','irum','nai']))

# NaN (결측치)
frame2 = pd.DataFrame(data, columns=['irum','nai','juso','tel'], 
                            index=['a','b','c','d','e'])# index 추가
print(frame2) 
# 칼럼의 값 수정.
frame2['tel']='111-1111'
print(frame2)
# 시리즈로 값 넣기 ** 시리즈와 데이터프레임과의 상관관계 중요 **
val = pd.Series(['222-2222','333-3333','444-4444'], index=['b','c','e'])
print(val)
# tel열을 완전히 덮어쓰기, 없는index값은 nan으로 들어옴.
frame2['tel'] = val
print(frame2)
print()

print(frame.T) # 전치
print()

print(frame2.values) # 결과는 list type이다.
# 인덱싱
print(frame2.values[0,1])
#슬라이싱
print(frame2.values[0:2])

# 행/열삭제
frame3 = frame2.drop('d') # 'd'행 삭제 == # frame3 = frame2.drop('d', axis=0)
print(frame3)
frame4 = frame2.drop('tel',axis=1) # 'tel' 열삭제
print(frame4)

# 정렬
print('-'*100)
print(frame2)
print(frame2.sort_index(axis=0, ascending=False))   # 행 단위 정렬
print(frame2.sort_index(axis=0, ascending=True))    # 열 단위 정렬

# 사전순으로 순서를 정함.
print(frame2.rank(axis=0))
# 칼럼안의 중복데이터를 카운트함.
counts = frame2['juso'].value_counts()
print(counts)

# 문자열 자르기
data = {
    'juso':['강남구 역삼동','중구 신당동','강남구 대치동'],
    'inwon':[23, 25, 15]
}
fr = pd.DataFrame(data)
print(fr)

# 자르기
# result1 = Series(x.split()[0] for x in fr.juso) # type tuple
# result1 = Series((x.split()[0] for x in fr.juso)) # type tuple
result1 = Series([x.split()[0] for x in fr.juso])# 공백을 구분자로 문자열을 분리함.
result2 = Series([x.split()[1] for x in fr.juso]) 
print(result1)
print(result2)
print(result1.value_counts())
