'''
힙(Heap)
    모든 노드가 특정한 순서를 유지하며 구성된 완전 이진 트리 형태의 자료구조다.
    힙은 두 가지 주요한 특성을 가진다.
        1) 구조적 특성 (Structural Property) : 항상 완전 이진 트리(Complete Binary Tree)의 형태를 유지한다.
            즉, 왼쪽에서 오른쪽으로 차례대로 채워지며, 마지막 레벨을 제외하고는 노드가 모두 채워진 형태다.
        2) 순서적 특성 (Heap Order Property) : 각 부모 노드와 자식 노드 사이의 값의 크기 관계에 따라 두 가지 유형
            이 존재한다.

    ~ 최대 힙 (Max Heap)
        - 부모 노드 ≥ 자식 노드
        - 루트에 가장 큰 값이 위치
        - 예: 우선순위가 높은 작업을 먼저 처리해야 할 때 사용
    ~ 최소 힙 (Min Heap)
        - 부모 노드 ≤ 자식 노드
        - 루트에 가장 작은 값이 위치
        - 예: 가장 작은 값부터 처리할 필요가 있을 때 사용 (예: 다익스트라 알고리즘)
'''
import heapq # 기본이 Min Hea

heap = []
heapq.heappush(heap, 30)
heapq.heappush(heap, 10)
heapq.heappush(heap, 20)
print("현재 힙 상태 :", heap) # 내부적으로 heap구조가 유지
#  [10, 30, 20]

# 최소값 꺼내기
print('가장 작은 값 :', heapq.heappop(heap)) # 10
print("남은 Heap :", heap)  # [20, 30]

print('가장 작은 값 :', heapq.heappop(heap)) #  20
print("남은 Heap :", heap)  # [30]


# Max Heap
heap = []
heapq.heappush(heap, -30) # Max Heap으로 사용하기 위해 - 를 붙이는 트릭 사용
heapq.heappush(heap, -10)
heapq.heappush(heap, -20)
print("현재 힙 상태 :", heap) # 내부적으로 heap구조가 유지
#  [-10, -30, -20]

# 최대값 꺼내기
print('가장 큰 값 :', -heapq.heappop(heap)) # 30
print("남은 Heap :", heap)  # [-10, -20]

print('가장 큰 값 :', -heapq.heappop(heap)) #  20
print("남은 Heap :", heap)  # [-10]