"""
list 안에 들어있는 자료를 오름차순 정렬
Bubble 정렬(bubble Sort)
    이웃한 요소와 비교
    [2, 4, 5, 1, 3]
    [2, 4, 5, 1, 3]
    [2, 4, 5, 1, 3]
    [2, 4, 1, 5, 3]
    [2, 4, 1, 3, 5]
    ....
"""
print("-"*30," 5) Bubble(Bubble Sort) ","-"*30)
print('[알고리즘을 위한 코드]')
def bubble_sort(a):
    n = len(a)
    
    # 정렬이 완료 될때 까지 작업을 반복함.
    while True:
        # 자료를 앞뒤 변경 여부
        change = False

        for i in range(0, n-1):
            # 앞이 뒤보다 크면
            if a[i] > a[i+ 1]:
                print(a)
                # 자리 바꾸기
                a[i], a[i+1] = a[i+1], a[i]
                # 바뀌었음을 기록함.
                change = True

        if change == False:
            return

d = [2, 4, 5, 1, 3]
bubble_sort(d)
print(d)