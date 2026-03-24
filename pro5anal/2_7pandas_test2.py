"""
1) human.csv 파일을 읽어 아래와 같이 처리하시오.
- Group이 NA인 행은 삭제
- Career, Score 칼럼을 추출하여 데이터프레임을 작성
- Career, Score 칼럼의 평균계산


참고 : strip() 함수를 사용하면 주어진 문자열에서 양쪽 끝에 있는 공백과 \n 기호를 삭제시켜 준다. 
그래서 위의 문자열에서 \n과 오른쪽에 있는 공백이 모두 사라진 것을 확인할 수 있다. 
주의할 점은 strip() 함수는 문자열의 양 끝에 있는 공백과 \n을 제거해주는 것이지 중간에 
있는 것까지 제거해주지 않는다.

2) tips.csv 파일을 읽어 아래와 같이 처리하시오.
- 파일 정보 확인
- 앞에서 3개의 행만 출력
- 요약 통계량 보기- 흡연자, 비흡연자 수를 계산  : value_counts()
- 요일을 가진 칼럼의 유일한 값 출력  : unique()
결과 : ['Sun' 'Sat' 'Thur' 'Fri']
"""
import pandas as pd
import numpy as np

print("-"*15," human","-"*15)
"""
1) human.csv 파일을 읽어 아래와 같이 처리하시오.
- Group이 NA인 행은 삭제
 Career, Score 칼럼을 추출하여 데이터프레임을 작성
- Career, Score 칼럼의 평균계산
"""
human_df = pd.read_csv("human.csv", sep='\s*,\s*', skipinitialspace=True) # , 앞뒤로 공백제거!
# print(human_df.columns)
print(human_df)
print(human_df.head(5))
# print(human_df['Group'])
human_df = human_df.dropna(subset=['Group'])
human_df2 = human_df[['Career','Score']]
print(human_df2.mean(axis=0))
print()

print("-"*15," tips","-"*15)
"""
2) tips.csv 파일을 읽어 아래와 같이 처리하시오.
- 파일 정보 확인
- 앞에서 3개의 행만 출력
- 요약 통계량 보기- 흡연자, 비흡연자 수를 계산  : value_counts()
- 요일을 가진 칼럼의 유일한 값 출력  : unique()
결과 : ['Sun' 'Sat' 'Thur' 'Fri']
"""
tips = pd.read_csv('tips.csv', sep='\s*,\s*')
print(tips.head(3))
print(tips.value_counts(subset=['smoker']))
print(pd.unique(tips['day']))