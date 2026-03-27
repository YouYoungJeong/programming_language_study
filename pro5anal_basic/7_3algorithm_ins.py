"""
list 안에 들어있는 자료를 오름차순 정렬
삽입 정렬(Insertion Sort)
    앞에서부터 하나씩 꺼내서 자기자리 찾아 끼워 넣는 정렬
    https://blog.kakaocdn.net/dna/pY2WC/btrGWzOvGga/AAAAAAAAAAAAAAAAAAAAAEyNjJnE5fD7Q6Yw0_cMiGy2CcAkpiNy9W5Y6HvCtsOX/img.gif?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1774969199&allow_ip=&allow_referer=&signature=cDxlM2%2FnvVvRz83QV%2BDGwQ78%2B%2FA%3D
"""
print("-"*30," 2) 삽입정렬(Insertion Sort) - O(n^2) ","-"*30)
print('[1) 이해를 위한 코드 - pop()사용]')
# (데이터, 벨류)
def find_ins_idx(r, v):
    for i in range(0, len(r)):
        # v값이 i번 위치값 보다 작으면 i값을 반환
        if v < r[i]:
            return i
    
    # 적정한 삽입 위치를 못 찾은 경우 맨 뒤에 삽입
    return len(r)

d = [2, 4, 5, 1, 3]
# 값들의 위치 확인
# print(find_ins_idx(d, 1)) # 0
# print(find_ins_idx(d, 2)) # 1
# print(find_ins_idx(d, 5)) # 5
# print(find_ins_idx(d, 4)) # 2

def ins_sort(a):
    result = []
    while a:
        value = a.pop(0)
        ins_idx = find_ins_idx(result, value)
        result.insert(ins_idx, value)
        print("result : ", result, "/ a : ", a)
        print()
    return result
d = [2, 4, 5, 1, 3]
print(ins_sort(d))
print()
# =======================================================================
print('[2) 알고리즘을 위한 코드]')
# 일반 알고리즘.
def ins_sort2(a):
    n = len(a)
    # 두번째 값(인덱스[1]) 부터 마지막까지 차례대로 '삽입할 대상 선택'
    for i in range(1, n):
        key = a[i]  # i번 위치에 있는 값 key에 저장.
        print(f'\n--------{key} 값 key에 저장---------\n{a}\n')
        j = i - 1   # j를 i바로 왼쪽 위치로 저장

        # 값 비교 시작
        while j >= 0 and a[j] > key:
            print(a)                
            a[j + 1] = a[j]         # 삽입 할 공간이 생기도록 값을 우측으로 밀어넣음
            j -= 1                  # 그 다음 왼쪽으로 이동하면서 다시 비교 작업 반복
        a[j + 1] = key              # 찾은 삽입 위치에 key값을 넣음 

d = [2, 4, 5, 1, 3]
ins_sort2(d)
print(d)