"""
list 안에 들어있는 자료를 오름차순 정렬
"""
print("-"*30," 1) 선택정렬(Selection Sort) - O(n^2) ","-"*30)
# 1) 선택정렬(Selection Sort) 입력값이 크면 시간도 오래걸리지만 이해하기가 쉽다.
# 이해 위주
d = [2, 4, 5, 1, 3]
d2 = [2, 4, 5, 3]
print('[1) 이해를 위한 코드 - pop()사용]')
def find_min_index(a):
    # 전체 갯수 확인
    n = len(a)
    # 제일 작은값 위치 찾기
    min_idx = 0 # 1번째 값 인덱스 미리 넣고
    for i in range(1, n):
        # a[i]번째 값이 a[min_idx]값보다 작으면 min_idx인덱스 값 i 로바꾸고 다 돌면 인덱스값 출력
        if a[i] < a[min_idx]:
            min_idx = i
    return min_idx # 제일 작은 값의 인덱스 값을 반환

print(find_min_index(d))
print(find_min_index(d2))

def select_sort(a):
    # a의 자료가 정렬된 값을 넣기위한 result list 기억장소 하나 더 추가됨.
    result = []
    # while - list(a)값이 사라질때까지 반복
    while a:
        min_idx = find_min_index(a)     # 제일 작은값의 인덱스를 저장
        value = a.pop(min_idx)          # 인덱스값을 뽑기
        result.append(value)            # pop으로 뽑은 값 result list로 저장.
    return result

print(select_sort(d))
print(select_sort(d2))
print()
# =======================================================================
print('[2) 알고리즘을 위한 코드]')
# 일반 알고리즘 - 기억장소 절약 효과도 있음
# O(n**2) 
def select_sort2(a):
    n = len(a)                  # 전체 크기 카운트
    for i in range(0, n - 1):   # 0부터 n-2회 반복
        min_idx = i
        for j in range(i + 1, n): # 다음 값 출력 
            if a[j] < a[min_idx]: # 다음 값 비교 
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i] # 찾은 최소값을 i번과 위치 변경(최소값 앞으로 밀기)

d = [2, 4, 5, 1, 3]
d2 = [2, 4, 5, 3]
select_sort2(d)
print(d)
select_sort2(d2)
print(d2)
print()