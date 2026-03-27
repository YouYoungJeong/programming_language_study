"""재색인
시리즈, 데이터프레임"""
from pandas import Series, DataFrame
import numpy as np

# 시리즈의 재색인
data = Series([1,3,2], index = (1, 4, 2))
print(data)
# reindex
data2 = data.reindex((1,2,4))
print(data2)

# 재색인시 값을 채워넣음
print("-"*15," 재색인 값 채우기 ","-"*15)
data3 = data2.reindex([0,1,2,3,4,5])
print(data3)
# fill_value : 대응값이 없는 인덱스에 특정값으로 채움.
data3 = data2.reindex([0,1,2,3,4,5], fill_value=777)
print(data3)
print()

# 방식method NaN 앞값으로  NaN을 채움 : 첫번째 값이 없을경우 NaN
data3 = data2.reindex([0,1,2,3,4,5], method='ffill')
print(data3)
data3 = data2.reindex([0,1,2,3,4,5], method='pad')
print(data3)

# 방식method NaN 뒤값으로  NaN을 채움 : 마지막 값이 없을경우 NaN
data3 = data2.reindex([0,1,2,3,4,5], method='bfill')
print(data3)
data3 = data2.reindex([0,1,2,3,4,5], method='backfill')
print(data3)

# DataFrame : bool처리
print("-"*15," DataFrame : bool처리 ","-"*15)
df = DataFrame(np.arange(12).reshape(4, 3),
                index=['1월','2월','3월','4월'],
                columns=['강남','강북','서초'])
print(df)
print(df['강남'])
# 조건을 주면 True, False값이 출력
print(df['강남'] > 3)
# df안에 넣으면 조건이 참인 경우만 출력
print(df[df['강남'] > 3])

# 조건 데이터에 숫자 적용
print(df < 3)
df[df < 3] = 0
print(df)

# 슬라이싱 관련 메소드 : loc() : 라벨지원 , iloc() : 숫자만 지원
print("-"*15," 슬라이싱 관련 메소드 : loc(), iloc() ","-"*15)
print(df.loc['3월', :]) #3월행의 모든 열 시리즈로 반환. 가독성이 좋아
print(df.loc[:'2월'])
print(df.loc[:'2월', ['서초']]) # 라벨지원.
print()
# iloc() : 숫자만 지원 2행의 모두 출력
print(df.iloc[2])
print(df.iloc[2,:])
print(df.iloc[:3]) # 3행 미만
print(df.iloc[:3, 2]) # 3행 미만 2열만
print(df.iloc[:3, 1:3])