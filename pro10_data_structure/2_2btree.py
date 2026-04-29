'''
이진트리(Binary Tree)
    tree중 자식이 2 이하인 tree
    이진트리 순회는 DFS(깊이 우선 탐색) 기반 -> 재귀가 필요하다.
    이진트리로 정렬을 할 수 있다. - Binary Serach Tree 사용
        중위순회는 BST정렬이 가능하다

    노드 방문 방법 3가지 
    pre-order(전위)  : Root -> L -> R
    in-order(중위)   : L -> Root -> R
    post-order(후위) : L -> R -> Root
'''

tree = {
    'A' : ('B', 'C'),
    'B' : ('D', 'E'),
    'C' : (None, None),
    'D' : (None, None), # leaf
    'E' : (None, None)  # leaf
}

# 전위 순회(Pre-Order)
def preOrder(node):
    if node is None:
        return
    print(node, end=' ')
    left, right = tree[node]    # Root
    preOrder(left)  # 재귀함수  # Left
    preOrder(right) # 재귀함수  # R


# 중위 순회(In-Order) - BST(Binary Search Tree)정렬이 가능
def inOrder(node):
    if node is None:
        return
    left, right = tree[node]
    inOrder(left)  # 재귀함수      # L
    print(node, end=' ')           # Root
    inOrder(right) # 재귀함수      #  R

# 후위 순회(Post-Order)
def postOrder(node):
    if node is None:
        return
    left, right = tree[node]
    postOrder(left)  # 재귀함수   # L
    postOrder(right) # 재귀함수   # R
    print(node, end=' ')         # Root

print("전위 순회 결과 :")
preOrder('A') # A B D E C 
print()

print("중위 순회 결과 :")
inOrder('A') # D B E A C
print()

print("후위 순회 결과 :")
postOrder('A') # D E B C A 