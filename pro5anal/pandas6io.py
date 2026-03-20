import pandas as pd
import numpy as np

items = {'apple' : {'count':10, 'price':1500},
        'orage' : {'count':5, 'price':800}
        }
df = pd.DataFrame(items)
print(df)

# DataFrame 저장
# 클립보드로 저장
df.to_clipboard() # 메모장 붙여넣기로 확인 가능
print(df.to_html())
print(df.to_json()) # 나중에 Ajax사용할때 좋겠다.

# csv로 저장
df.to_csv('result1.csv', sep=',') # 구분자를 만들어줘서 넣기
df.to_csv('result2.csv', sep=',', index=False) 
df.to_csv('result3.csv', sep=',', index=False, header=False) 
print()
df2 = df.T
print(df2)
df2.to_csv('result4.csv', sep=',', index=False, encoding='utf-8-sig') 
# utf-8-sig : 엑셀까지 생각하면sig붙여야함
redata = pd.read_csv('result4.csv')
print(redata)

print()

print("-"*15," 엑셀관련","-"*15)
""" 엑셀 관련: 현업에서 많이 사용함. python 과 엑셀을 잘쓰면 정말 편해져"""
df3 = pd.DataFrame({
    'name':['Alice','Bob','Oscar'],
    'age' : [24, 32, 29],
    'city':['seoul','suwon','incheon']
})
print(df3)

# 엑셀로 저장 i/o
df3.to_excel('result.xlsx', index=False, sheet_name='work1')

# 엑셀 읽기 i/o - 엑셀창 끄고 사용 가능 켜져있으면 denied
exdf = pd.ExcelFile('result.xlsx')
print(exdf.sheet_names)

# 시트 하나씩 가져오기
df4 = exdf.parse('work1')
print(df4)