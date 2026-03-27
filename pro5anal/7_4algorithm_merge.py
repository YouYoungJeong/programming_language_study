"""
list 안에 들어있는 자료를 오름차순 정렬
병합정렬(Merge Sort)
    list 자료를 요소가 한개씩 남을 때까지 반복해서 반으로 나눔
    분할된 리스트를 정렬상태 유지 하며 하나로 병합함.
"""
print("-"*30," 3) 병합정렬(Merge Sort) - 재귀사용 ","-"*30)
print('[1) 이해를 위한 코드 - pop()사용]')

def merge_sort(a):
    n = len(a)

    if n <= 1:
        return a
    
    mid = n //2 # 중간을 기준으로 두 그룹으로 분할

    # 함수는 독립적인 공간을 갖음. (재귀를 이해하기위해 꼭 알아야함.)
    # 아래의 g1, g2는 불간섭 (메모리가 다름)
    # 재귀
    g1 = merge_sort(a[:mid])    # 중간 앞
    # print('g1 : ', g1)
    g2 = merge_sort(a[mid:])    # 중간 뒤
    # print('g2 : ', g2)

    # 여러개로 분리된 두그룹들을 하나로 만들기
    result = []
    while g1 and g2:
        print(g1[0], " ",g2[0])
        if g1[0] < g2[0]:
            result.append(g1.pop(0))
        else:
            result.append(g2.pop(0))
        print('result : ', result)

    # g1과 g2중 소진된 것은 스킵
    while g1:
        result.append(g1.pop(0))
    while g2:
        result.append(g2.pop(0))

    return result


d = [6, 8, 3, 1, 2, 4, 7, 5]
print(merge_sort(d))

# =======================================================================
print('[2) 알고리즘을 위한 코드]')
# 재귀호출이 정렬된 리스트를 반환.
# 병합도 새 리스트를 만들어 반환.
# 원본 리스트는 그대로이고 정렬된 결과는 새 리스트에 저장.
# 재귀를 사용할 때 데이터의 양이 많아지면 많아질 수록 메모리가 많이 필요함.
def merge_sort2(a):
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    # 길이가 1이 될때 까지 재귀 호출해 계속 반으로  나누기.
    left = merge_sort2(a[:mid])
    right = merge_sort2(a[mid:])

    resrult = []
    i = j = 0

    # 병합(merge)
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            resrult.append(left[i])
            i += 1
        else:
            resrult.append(right[j])
            j += 1
    
    # 남은 요소 추가
    resrult += left[i:]
    resrult += right[j:]
    return resrult


d = [6, 8, 3, 1, 2, 4, 7, 5]
sorted_d = merge_sort2(d)
print(sorted_d)