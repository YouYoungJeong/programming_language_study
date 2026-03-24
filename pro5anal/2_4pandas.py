"""
DataFrame 재 구조화 (열을 행으로, 행을 열로 이동)
stack = 인덱스를 기준으로 칼럼을 인덱스로 가져옴
unstack = 스텍결과를 원복할때 사용함.행을 열로 이동
cut() = 연속형 데이터를 범주화 cut(컷할 대상, 컷할 범주)
qcut() =  구간을 상수로정의해서 사용, 데이터의 량이 많은 경우 사용하기 좋음
# () : 초과 미만,  [] : 이상 이하
agg() = 범주의 그룹별로 연산이 가능

"""
import pandas as pd
import numpy as np

df = pd.DataFrame(1000 + np.arange(6).reshape(2,3), index=['대전','서울'],
                    columns=['2020','2021','2022'])
print(df)
print()

print("-"*15,"stack, unstack","-"*15)
"""stack, unstack"""
df_row = df.stack() # 인덱스를 기준으로 칼럼의 열을 행으로 변환
print(df_row)
print()
# 원복
df_col  = df_row.unstack() # 행을 열로 이동
print(df_col)
print()

print("-"*15,"범주화","-"*15)
""" 범주화 cut(), qcut()"""
price = [10.3, 5.5, 7.8, 3.6]       # 실수, 연속형 데이터를 범주형으로 바꿀 수 있다.
cut = [3, 7, 9, 11] # 구간 기준값
result_cut = pd.cut(price, cut) # 연속형 데이터를 범주화 cut(컷할 대상, 컷할 범주)
print(result_cut)
# () : 초과 미만=> (a,b] = a<x<=b,  [] : 이상 이하
print(pd.Series(result_cut).value_counts())

print()
datas = pd.Series(np.arange(1, 1001))
# 데이터 값이 많을때 앞에만 보거나, 뒤에만 볼 때 사용함.
print(datas.head(5))
print(datas.tail(5))
result_cut2 = pd.qcut(datas, 3) # 3개로 분류해줘 -  볌주화 대상이 많을경우 사용
print(result_cut2)
print(pd.Series(result_cut2).value_counts())
print()

print("-"*15," [소계] 범주의 그룹별 연산","-"*15)
""" 범주의 그룹별 연산 agg(), apply()"""
group_col = datas.groupby(result_cut2, observed=True) # 객체 생성
# observed=True 데이터가 있는 경우에만 작업, false를 하면 데이터가 없는 경우에도 작업함
# print(group_col) 
print(group_col.agg(['count','mean','std','min']))
#agg(['count','mean','std','min'])) -'count','mean','std','min'.. : 구하고싶은 함수 적으면 된다.

# agg 대신 사용사 함수를 작성
def summaryFunc(gr):
    return {'coungt':gr.count(),
            'mean':gr.mean(),
            'std':gr.std(),
            'min':gr.min()
            }
print(group_col.apply(summaryFunc)) # **apply() : 함수를 실행하는 함수
print(group_col.apply(summaryFunc).unstack()) # agg()과 모양 똑같이 만들기
print()

print("-"*15,"데이터 프레임 객체 병합","-"*15)
""" 데이터 프레임 객체 병합 merge(), concat()"""
df1 = pd.DataFrame({'data1':range(7),'key':['b','b','a','c','a','a','b']})
print(df1)
df2 = pd.DataFrame({'key':['a','b','d'],'data2':range(3)})
print(df2)
print()
print(pd.merge(df1, df2, on='key')) # 공통으로 가지고 있는 key를 기준으로 병합(교지합, inner join)
print(pd.merge(df1, df2, on='key', how='inner')) # how='inner' = inner join 위랑 동일
print()
print(pd.merge(df1, df2, on='key', how='outer')) # how='inner' = 기준키를 가지고 full outer join함.
print()
print(pd.merge(df1, df2, on='key', how='left')) # how='left' = 기준키를 가지고 left outer join함.
print()
print(pd.merge(df1, df2, on='key', how='right')) # how='right' = 기준키를 가지고 right outer join함.
print()

# 성격이 같은 상태에서 공통 칼럼이 없는 경우 : df1 vs df3
df3 = pd.DataFrame({'key2':['a','b','d'], 'data2':range(3)})
print(df3)
print(df1)
print(pd.merge(df1, df3, left_on='key', right_on='key2')) # 직접 칼럼을 정해줌inner join
print()

print(pd.concat([df1, df3], axis=0)) # 행단위 자료 이어붙이기
print(pd.concat([df1, df3], axis=1)) # 열단위 자료 이어붙이기

print()

print("-"*15,"pivot_table()","-"*15)
""" 
pivot과 groupby 명령의 중간적 성격 pivot_table()
pivot(회전, 중심점) : 데이터 열 중에 두개의 열(key)을 사용해 데이터의 행열을 재구성, 
                    집계 메소드중 하나
set_index(['city','year']) + .unstack()도 가능
"""
data = {'city':['강남','강북','강남','강북'],
        'year':[2000, 2001, 2002, 2002],
        'pop':[3.3, 2.5, 3.0, 2.0]} # 수치데이터가 필수다.
df = pd.DataFrame(data)
print(df)
print()
print(df.pivot(index='city',columns='year',values='pop')) # 행은 city, 칼럼은 year, 조건은 pop
print(df.pivot(index='year',columns='city',values='pop'))
print()
print(df.set_index(['city','year']).unstack()) # set_index : 기존 행 인덱스를 제거하고 첫번째 열 인덱스 설정
print()
# 요약통계를 피봇_테이블로
print(df['pop'].describe())
print(df.pivot_table(index=['city'])) # 인덱스별로 그룹화
print(df.pivot_table(index=['city'], aggfunc='mean')) #  aggfunc='mean' <- 별명 사용 권장
print(df.pivot_table(index=['city','year'], aggfunc=[len,'mean'])) # 그룹이 복수면 []
print(df.pivot_table(index=['city','year'], aggfunc=[len,'sum'])) # len건수
print(df.pivot_table(values='pop', index='city')) # 소계 함수를 아무것도 안쓰면 평균
print(df.pivot_table(values='pop', index='city', aggfunc=len))
print()

print(df.pivot_table(values='pop', index=['year'], columns=['city']))
print(df.pivot_table(values='pop', index=['year'], columns=['city'], margins=True))
# margins=True 행렬에 대한 평균 All이 붙음
print(df.pivot_table(values='pop', index=['year'], columns=['city'], margins=True, fill_value=0))
print()
hap = df.groupby(['city'])
print(hap)
print(hap.sum()) 
print()
# 위에 단계를 한번에 쓰려면 이렇게
print(df.groupby(['city']).sum())
print(df.groupby(['city']).mean())