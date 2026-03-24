"""
연산 기타 등등
"""
from pandas import Series, DataFrame
import numpy as np

s1 = Series([1,2,3], index=['a','b','c'])
s2 = Series([4,5,6,7], index=['a','b','d','c'])
print(s1)
print(s2)
# 더하기, 대응되는 수가 없으면(인덱스 불일치 시) NaN, 같은 인덱스 끼리 연산함.
print(s1 + s2)
print(s1.add(s2)) # pandas에서 numpy의 함수를 계승함.
print(s1 * s2)
print(s1.mul(s2))
print(s1 - s2)
print(s1.sub(s2))
print(s1 / s2)
print(s1.div(s2))
print()

# columns=list('kbs') - 칼럼이 한글자씩 짤려서 들어감
df1 = DataFrame(np.arange(9).reshape(3, 3), columns=list('kbs'), index=['서울','대전','부산']) 
df2 = DataFrame(np.arange(12).reshape(4, 3), columns=list('kbs'), index=['서울','대전','제주','광주']) 
print(df1)
print(df2)
# fillvalue는 못주고 df.add에 fillvalue는함수가 들어있음
print(df1+df2)
print(df1.add(df2, fill_value=0)) # NaN값 대신 0으로 채운후 연산에 참여(sub, mul, div도 가능)
print()

# NaN(결측값) 처리
print("-"*15," NaN(결측값) 처리 ","-"*15)
df = DataFrame([[1.4, np.nan], [7, -4.5], [np.nan, np.nan],[0.5, -1]],
                columns=['one','two'])
print(df)
print()

#NaN 값 확인 : Null값을 탐지 isnull() / notnull()
print(df.isnull()) # nan =True
print(df.notnull()) # nan =False
print()

# 결측값이 하나라도 있으면 지움 dropna() == dropna(how='any')
print(df.dropna()) 
print(df.dropna(how='any'))
print() 
# 그행에 모든 값이 nan인경우 다 지움.
print(df.dropna(how='all'))
print()
# one칼럼의 nan값을 지움 - 특정열의 nan이 있는 행 삭제 / subset=
df.dropna(subset=['one']) 
print()
print(df.dropna(axis='rows')) # nan이 포함된 행을 지움
print(df.dropna(axis='columns')) # nan이 포함된 열을 지움
print()


""" 
** 꼭 알아야 하는것 
    drop는 원본은 삭제 되지 X, 삭제된 결과가 새로운  DataFrame으로 만들어짐
    inplace=True라는 옵션이 들어가면 원본이 삭제됨.
    기본적으론 inplace=False - 원본유지
"""
print("-"*15," 꼭 알아야 하는것 inplace ","-"*15)
print(df)
imsi = df.drop(1) # 1행 삭제
print(imsi)
print(df)
# 원본을 삭제하는 옵션
print()
# df.drop(1, inplace=True)
# print(df)

print("-"*15," 계산관련 메소드 ","-"*15)
# 열의 합 - NaN은 연산에서 제외
print(df.sum()) 
print(df.sum(axis=0))
print(df.sum(axis=0, skipna=True))
# 열의 합 - NaN은 연산에서 참여
print(df.sum(axis=0, skipna=False))

# 행의 합 - NaN은 연산에서 제외
print(df.sum(axis=1))
print()

#요약 통계량 출력 describe()
print(df.describe())
words = Series(['봄','여름','가을','봄'])
print(words.describe())
print()

#구조 출력 info()
print(df.info())
