'''
Binary Search Tree(BST, 이진 탐색 트리)
    중위 순회(In-Order) - BST(Binary Search Tree)
    오름차순 정렬 가능
    각 노드 기준으로 left < node < rigth
    왼쪽 서브트리 : 현재 노드보다 작은 값
    오른쪽 서브트리 : 현재 노드보다 큰 값
    입력 순서에 따라 트리모댱이 달라짐
    이진트리는 구조만 있으나 BST구조 + 정렬 규칙이 있다.
    중위 순회(L->Root(현재노드)->R)를 하면 오름차순 정렬
'''
# BST 노드 정의
class Node:
    def __init__(self, key):
        self.key = key # 노드가 저장하는값
        self.left = None # 왼쪽 자식 노드(더 작은 값들이 저장)
        self.right = None # 왼쪽 자식 노드(더 큰 값들이 저장)

# BST 삽입
def insert(root, key):
    if root is None: # 현재 위치가 비워져있으면 
        return Node(key)    # 새 Node생성
    
    if key < root.key: # 넣을 값이 현재 노드보다 작으면
        root.left = insert(root.left, key) # 왼쪽 서브트리에 재귀적으로 삽입한다.
    else:
        root.right = insert(root.right, key)
    
    return root

# 중위 순회(정렬 결과 생성)
def inorder(root, result):
    if root is None: # 더 내려갈 노드가 없다면 함수 탈출
        return 
    inorder(root.left, result)  # 왼쪽 노드(작은 값들) 방문
    result.append(root.key)     # 현재 노드값 추가
    inorder(root.right, result) # 오른쪽 노드(큰 값들) 방문

values = [5, 3, 7, 2, 4, 9]
root = None # 아직 tree가 없음
for v in values:
    root = insert(root, v) # BST에 삽입하고 Root

# BST정렬 결과
sorted_result = []
inorder(root, sorted_result)
print('결과 :', sorted_result)
'''
inorder(5)
    -> inorder(3)
        -> inorder(2)
            -> inorder(Node) <- 멈춤
            이제 하나씩 올라오면서 처리
                (2 처리) result.append(2) <- result = [2] 
                (3 처리) result.append(3) <- result = [2, 3] 
        -> inorder(4)
            ->inorder(None) <-멈춤
                (4 처리) result.append(4) <- result = [2, 3, 4] 
                (5 처리) result.append(5) <- result = [2, 3, 4, 5] 
        -> inorder(7)
            ->inorder(None) <-멈춤
                (7 처리) result.append(7) <- result = [2, 3, 4, 5, 7] 
        -> inorder(9)
            ->inorder(None) <-멈춤
                (9 처리) result.append(9) <- result = [2, 3, 4, 5, 7, 9]

'''