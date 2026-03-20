"""
열 구성 정보
    Survived : 0 = 사망, 1 = 생존
    Pclass : 1 = 1등석, 2 = 2등석, 3 = 3등석
    Sex : male = 남성, female = 여성
    Age : 나이
    SibSp : 타이타닉 호에 동승한 자매 / 배우자의 수
    Parch : 타이타닉 호에 동승한 부모 / 자식의 수
    Ticket : 티켓 번호
    Fare : 승객 요금
    Cabin : 방 호수
    Embarked : 탑승지, C = 셰르부르, Q = 퀸즈타운, S = 사우샘프턴
1) 데이터프레임의 자료로 나이대(소년, 청년, 장년, 노년)에 대한 생존자수를 계산한다.
cut() 함수 사용
bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]
2) 성별 및 선실에 대한 자료를 이용해서 생존여부(Survived)에 대한 생존율을 피봇테이블 형태로 작성한다. 
df.pivot_table()
index에는 성별(Sex)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
출력 결과 샘플1 :       

pclass	1	2	3
sex			
female	0.968085	0.921053	0.500000
male	0.368852	0.157407	0.135447
index에는 성별(Sex) 및 나이(Age)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
출력 결과 샘플2 : 위 결과물에 Age를 추가. 백분율로 표시. 소수 둘째자리까지.    예: 92.86
"""
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', False) # 모든 행 출력

titanic = pd.read_csv('titanic_data.csv')
print(titanic.head(5))
print()

print("-"*15," titanic-1","-"*15)
""" 
1) 데이터프레임의 자료로 나이대(소년, 청년, 장년, 노년)에 대한 생존자수를 계산한다.
cut() 함수 사용
"""
age =titanic['Age']
bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]
titanic['나이대'] = pd.cut(age, bins=bins, labels=labels)
result = titanic.groupby('나이대', observed=True)['Survived'].sum()
result= result.reset_index()
result.columns=['나이대','생존자수']
print(result)



print("-"*15," titanic-2_1","-"*15)
""" 
2) 성별 및 선실에 대한 자료를 이용해서 생존여부(Survived)에 대한 생존율을 피봇테이블 형태로 작성한다. 
df.pivot_table()
index에는 성별(Sex)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
출력 결과 샘플1 :       
pclass	1	2	3
sex			
female	0.968085	0.921053	0.500000
male	0.368852	0.157407	0.135447
"""
# survi = titanic[['Survived','Pclass','Sex']]
# survi_piv = survi.pivot_table(index=['Sex'], columns=['Pclass'], aggfunc='mean')
# print(survi_piv)
pivot1 = titanic.pivot_table(
    values='Survived',
    index='Sex',
    columns='Pclass',
    aggfunc='mean',
    observed=True)
print(pivot1)
print()


"""
index에는 성별(Sex) 및 나이(Age)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
출력 결과 샘플2 : 위 결과물에 Age를 추가. 백분율로 표시. 소수 둘째자리까지.    예: 92.86
"""
print("-"*15," titanic-2_2","-"*15)
pivot2 = titanic.pivot_table(
    values='Survived',
    index=['Sex', '나이대'],
    columns='Pclass',
    aggfunc='mean'
)
pivot2 = (pivot2 * 100).round(2)
print(pivot2)





