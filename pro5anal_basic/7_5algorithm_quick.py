"""
list 안에 들어있는 자료를 오름차순 정렬
Quick 정렬(Quick Sort)
    하나의 기준점을 중심으로 작은값과 큰값을 나눠서 각각 정렬 후
    마지막에 이어 붙이는 방법
        방법)
        [6, 8, 3, 1, 2, 4, 7, 5]
        group1 = 기준값보다 작은 그룹   :[3, 1, 2, 4]  
                                        -> 기준값 :4 
                                        g1:[3, 1, 2], g2:[X] ....
                                            -> 다 쪼갠후. g1 + 기본값 + g2 ....
        기준값(보통은 제일 마지막 값을 사용) : 5
        group2 = 기준값보다 큰 그룹     :[6, 8, 7]
"""
print("-"*30," 4) Quick정렬(Quick Sort) - 재귀 사용 ","-"*30)
print('[1) 이해를 위한 코드]')
def quick_sort(a):
    n = len(a)
    # 종료 조건 생성 - 자르다가 길이가 1이 되면 a값을 반환
    # 재귀는 호출되는 동안 수행하지 않고 return을 만나면 그때 수행시작
    if n <= 1:
        return a

    # 기준값 생성(편의상 보통 맨뒤값을 많이 사용함)
    pivot = a[-1]

    # 그룹 저장공간 생성
    g1 = [] # 기준값 보다 작은 그룹
    g2 = [] # 기준값 보다 큰 그룹

    # 반복문 시작(마지막 값은 뽑았으니 마지막 값은 참여X)
    for i in range(0, n-1):
        if a[i] < pivot:
            g1.append(a[i])
        else:
            g2.append(a[i])

    # 최종적으로 나눠진 그룹 확인 
    print('g1 : ', g1)
    print('g2 : ', g2)

    # 재귀 리턴
    return quick_sort(g1) + [pivot] + quick_sort(g2)


d = [6, 8, 3, 1, 2, 4, 7, 5]
print(quick_sort(d))
print()
# =======================================================================
print('[2) 알고리즘을 위한 코드]')
def quick_sort_sub(a, start, end):
    # 종료 조건 생성 -> 정렬 대상이 1개 이하이면 정렬 정지(끝 인덱스 - 시작인덱스=0)
    if end - start <= 0:
        return
    
    pivot  = a[end]
    i = start
    
    for j in range(start, end):
        if a[j] <= pivot:
            # 자리 바꾸기
            a[i], a[j] = a[j], a[i]
            print(f"a[i], a[j] = a[j], a[i] => {a[i]}, {a[j]}, = {a[j]}, {a[i]}")
            i += 1


    # pivot 자기자리에 넣기
    a[i], a[end] = a[end], a[i]
    print()
    print("a[i], a[end] = a[end], a[i] => ",a[i], a[end], '=' ,a[end], a[i])

    # 기준값보다 작은 그룹 재귀로 다시 정렬 - 왼쪽정렬
    quick_sort_sub(a, start, i-1)

    # 기준값보다 큰 그룹 재귀로 다시 정렬 - 오른쪽 정렬 
    quick_sort_sub(a, i + 1, end)



def quick_sort2(a):
    quick_sort_sub(a, 0, len(a) -1) # (정렬 대상 data, 시작 인덱스, 끝 인덱스)


d = [6, 8, 3, 1, 2, 4, 7, 5]
# 메모리 하나만 사용
quick_sort2(d)
print(d)