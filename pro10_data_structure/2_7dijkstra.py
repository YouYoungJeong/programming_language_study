'''
https://cafe.daum.net/flowlife/SBU0/78

다익스트라(Dijkstra) 알고리즘
    pdf 내용을 코드로 구현

    휴리스틱(Heuristic)은 공부가 꼭 필요하다
'''
import heapq

INF = int(1e9) # 무한대 역할을 하는 값을 줌.

# 그래프 (인접 리스트 방식) - (Node번호-1, 비용) 형태로 저장
graph = [ # Node1은 0번째, Node2은 1번째...
    [(1, 2), (2, 5), (3, 1)], # Node1 [(Node2, 2), (Node3, 5), (Node4, 1)]   
    [(0, 2), (2, 3) ,(3, 2)], # Node2
    [(0, 5), (1, 3) , (3, 3),(4, 1), (5, 5)], # Node3
    [(0, 1), (1, 2) ,(2, 3), (4, 1)], # Node4
    [(2, 1), (3, 1) ,(5, 2)], # Node5
    [(2, 5), (4, 2)], # Node6
]

# 노드 갯수
n = 6
# 최단 거리 배열 초기값 엄청 큰값으로 초기화
distance = [INF] * n

def dijkstraFunc(start):
    # 우선순위 Queue(Heap)
    pq = []
    distance[start] = 0 # 초기값

    # (거리, node) 형태로 queue에 삽입
    heapq.heappush(pq, (0, start))

    while pq:
        dist, now = heapq.heappop(pq)

        if distance[now] < dist:
            continue

        # 현재 노드에서 갈 수 있는 모든 노드 탐색하기
        for next_node, cost in graph[now]:
            new_cost = dist + cost

            # 만약 새로운 경로가 더 짧으면 
            if new_cost < distance[next_node]:
                # 최단거리 갱신
                distance[next_node]=new_cost
                # heap에 삽입
                heapq.heappush(pq, (new_cost, next_node))




dijkstraFunc(0) # Node1에서 출발

# 각 노드 까지의 최단거리 출력
for i in range(n):
    print(f'Node{i + 1} 까지 최단거리 : {distance[i]}')