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