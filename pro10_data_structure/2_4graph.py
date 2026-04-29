'''
Graph(그래프)
    정점(Vertex, Node)과 정점을 연결하는 간선(Edge)의 집합으로 이루어진 자료구조다.
    핵심 포인트는 부모·자식 관계가 아니다, 위·아래 개념이 없다, 서로 자유롭게 연결될 수 있다
    Tree가 “조직도” = "계층구조", Root가 있음, 사이클(순환) 없음, 항상 연결
    Graph는 “지도·네트워크”구조, Root가 없음, 사이클(순환) 있음, 연결 비연결 모두 가능

용어 설명
- 노드(Node, 정점) : 데이터를 나타내는 점
- 엣지(Edge, 간선) : 노드 간의 연결 또는 관계
- 가중치(Weight) : 엣지에 숫자 값이 추가된 경우 (예: 거리, 비용 등)

** BFS(너비 우선 탐색)과 DFS(깊이 우선 탐색)에 대한 개념, 차이점, 작동 방식, 사용 사례 등을 설명
    DFS - 깊이 우선 탐색 방식 - 재귀함수 또는 스택으로 구현 -> 경로 추적, 백트래킹(Undo)에 적합
    BFS - 너비 우선 탐색 방식 - Queue로 구현  -> 최단거리 탐색에 적합(우리는 이게 중요)
    1   -   2
    |       |   DFS(1시작) : 1 -> 2 -> 4 -> 3
    3   -   4   BFS(1시작) : 1 -> 2 -> 3 -> 4
    +---------------------------------------------------------------------------+
    |항목        | DFS(깊이 우선 탐색)       | BFS(너비 우선 탐색)               |
    +---------------------------------------------------------------------------+
    |탐색 방향   | 한 방향으로 깊게          | 가까운 노드부터 넓게               |
    |자료 구조   | 스택(Stack) or 재귀       | 큐(Queue)                         |
    |구현 난이도 | 간단(재귀로 구현 쉬움)    | 큐 사용(조금 더 복잡)              |
    |주요 사용 예| 백트래킹, 경로 전체 탐색  | 최단거리, 레벨 기반 탐색           |
    |경로 보장   | 최단 경로 아님            | 최단 경로 보장(가중치 없음)        |
    +---------------------------------------------------------------------------+
'''
graph = {
    'A': ('B', 'C'),
    'B': ('D', 'E'),
    'C': ('F',),   # 원소 1개짜리 튜플
    'D': (),
    'E': (),
    'F': ()
}
'''
#           A
#        /    \ 
        B        C
#     /   \      |
    D       E    F
    '''
# DFS(깊이)
def dfsFunc(graph, start, visited):
    visited.append(start)   # 현재 노드 방문

    for next_node in graph[start]:
        if next_node not in visited:            # 아직 방문하지 않은 노드
            dfsFunc(graph, next_node, visited)  # 재귀로 방문

visited_dfs = []  # 방문 순서 저장 리스트
dfsFunc(graph, 'A', visited_dfs)
print('DFS 방문 순서 :', visited_dfs) 
# ['A', 'B', 'D', 'E', 'C', 'F']
# A -> B -> D -> (끝) -> E -> (끝) -> C -> F
# 방문 즉시 아래로 내려감 - 재귀(call stack)가 핵심

# BFS(넓이)
from collections import deque

def bfsFunc(graph, start):
    visited = [start] # 방문기록
    queue = deque([start]) # 큐 사용(FIFO)

    while queue:
        node = queue.popleft()              # 가장 먼저 들어온 노드를 꺼냄
        for next_node in graph[node]:       # 현재 노드와 이웃노드 확인
            if next_node not in visited:    # 방문이 안된 노드면
                visited.append(next_node)   # 방문 처리
                queue.append(next_node)     # 다음 탐색 대상으로 큐에 추가 
    # 큐에 처리할 노드가 남아있는 동안 반복이 끝나면
    return visited

visited_bfs = bfsFunc(graph,'A')  # 방문 순서 저장 리스트
print('BFS 방문 순서 :', visited_bfs) 
# ['A', 'B', 'C', 'D', 'E', 'F']
# 방문 즉시 큐에 쌓고 먼저 들어온 것부터 처리(거리<레벨> 개념이 생김)
