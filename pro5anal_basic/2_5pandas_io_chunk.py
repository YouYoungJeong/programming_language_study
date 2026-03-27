"""
pandas file 입출력
"""
import pandas as pd
import numpy as np

# csv파일을 읽어서 바로 df으로 만듦.
df = pd.read_csv('ex1.csv')
print(df, type(df)) # <class 'pandas.core.frame.DataFrame'>
print()

# csv를 하나로 읽어버림.
df = pd.read_table('ex1.csv')
print(df, type(df))

df = pd.read_table('ex1.csv', sep=',') # 구분자를 줘야함
print(df, type(df))

df = pd.read_table('ex1.csv', sep=',', skip_blank_lines=True) # 칼럼명, 데이터의 앞에
# skip_blank_lines=True  칼럼명, 데이터의 앞에 공백을 제거 하는 옵션
print(df, type(df))
print()

# github -> Row 로 호출, 웹상에서 바로 읽을 수 있다.
pd.set_option('display.max_columns', None) # Df의 모든 칼럼 표기 옵션.
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv')
print(df)
print()

# 데이터가 칼럼명이 없는 경우
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                header=None) 
print(df)
print()

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                header=None, skiprows=1 )  #첫번째 행을 스킵
print(df)
print()

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                header=None, 
                names=['a','b','c','d','e'])  # 칼럼 이름 주기
print(df)
print()

# txt파일 읽기
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt')
# print(df[:,0]) # csv로 읽어와서 안 짤림
df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt',
                sep='\s+') # 정규표현식 사용 -> \s+: 공백이 하나 이상
print(df.iloc[:,0])

df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt',
                sep='\s+',      # 정규표현식 사용 -> \s+: 공백이 하나 이상
                skiprows=[1, 3]) # skiprows=[1, 3](1행, 3행) 특정행을 읽지 않겠다 튜플, 리스트 가능
                                            
print(df)

# 구분자X, 공백X없는 경우 자리수를 읽는수 밖에 없음
# read_fwf는 widths=(자리수)를 줄 수 있다.
df = pd.read_fwf('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/data_fwt.txt',
                header=None,
                widths=(10, 3, 5), # 구분할 자리 수
                names=('date','name','price'),
                encoding='utf8') # 한글 깨짐 방지
                                            
print(df)
print(df.iloc[:,0])
print(df['date'])
print()

print("-"*15," 대량의 데이터를 부분씩 메모리로 읽어 처리 chunk ","-"*15)
""" chunk
큰덩어리를 작은 덩어리로 쪼개서 읽음.
대량의 데이터를 부분씩 메모리로 읽어 처리
대용량 자료 로딩시 초과 오류 발생 방지 : 메모리를 절약
스트리밍 방식(일부만 순차-시퀀스 처리)으로 읽음.
분산처리의 효과
여러번 반복해 읽어야 하므로 속도는 느리다.
"""
import time
n_rows = 10000
data = {
    'id':range(1, n_rows + 1),
    'name':[f'Student_{i}' for i in range(1, n_rows+ 1 )],
    'score1':np.random.randint(50, 101, size=n_rows),
    'score2':np.random.randint(50, 101, size=n_rows)
    }
df = pd.DataFrame(data)
print(df.head(5))
print(df.tail(3))
# 파일저장
csv_fname = 'students.csv'
df.to_csv(csv_fname, index=False)
print()

print("-"*15," csv 파일 읽기","-"*15)
start_all = time.time()
df_all = pd.read_csv(csv_fname)
average_all_1 = df_all['score1'].mean()
average_all_2 = df_all['score2'].mean()
# csv 전체 읽은 시간
time_all = time.time() - start_all
print('\n처리결과')
print('전체 한번에 처리 시간 : ', round(time_all, 7))
print()

#chunk로 읽기
chunk_size = 1000
total_socre1 = 0
total_socre2 = 0
total_count = 0
start_chunk_total = time.time()

# enumerate() 데이터에대한 index를 얻음
for i, chunk in enumerate(pd.read_csv(csv_fname ,chunksize=chunk_size)):
    start_chunk = time.time()
    # chunk 처리 중 첫번째 학생 정보는 출력
    first_student = chunk.iloc[0]
    print(f"Chunk {i + 1} 첫번째 학생:ID = {first_student['id']}, \
        이름 = {first_student['name']}",
        f"score1 = {first_student['score1']}, score2 = {first_student['score2']}")
    total_socre1 += chunk['score1'].sum()
    total_socre2 += chunk['score2'].sum()
    total_count += len(chunk)

    end_chunk = time.time()
    elapsed = end_chunk - start_chunk
    print(f'    처리 시간 : {elapsed:7f}') # 청크단위 처리시간
time_chuck_total = time.time() - start_chunk_total
ave_chuck1 = total_socre1/ total_count
ave_chuck2 = total_socre2/ total_count

print('\n처리결과')
print(f"전체 학생수 : {total_count}")
print(f"score1 총합 : {total_socre1}, 평균 : {ave_chuck1:3f}")
print(f"score2 총합 : {total_socre2}, 평균 : {ave_chuck2:3f}")
print(f'전체 한번에 처리 시간 : {time_all:7f}초')
print(f'청크로 처리한 총시간 : {time_chuck_total:7f}초')


# chunk 처리시간 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic') # 윈도우 기준 글씨체
labels = ['전체 한번에 처리', '청크로 처리']
times = [time_all, time_chuck_total]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, times, color=['skyblue','red'])
for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2, \
                bar.get_height(), f'{time_val:3f}초',\
                ha='center', va='bottom', fontsize=10)
plt.ylabel('처리시간(초)')
plt.grid(linestyle='--')
plt.tight_layout()
plt.show() #vscode는 꼭 써야 보임 jupyter는 안씀.