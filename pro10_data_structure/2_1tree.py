'''
Tree
    비선형 구조에서의 tree
    노드들이 나무가지 처럼 연결된 계측정 비선형 자료구조

    회사 조직도 구조 Tree로 표현
'''

company = {
    'CEO' : ['개발팀장', '기획팀장', '영업팀장'],
    '개발팀장' : ['백엔드', '프론트엔드'],
    '기획팀장' : ['서비스기획'],
    '영업팀장' : ['국내영업', '해외영업'],
    '백엔드' : [],      # leaf
    '프론트엔드' : [],   # leaf
    '서비스기획' : [],   # leaf
    '국내영업' : [],     # leaf
    '해외영업' : []      # leaf
}

# tree 구조 함수
def showTree(node, level):
    print(" " * level + ' - ' + node)
    
    # 현재 노드의 자식 출력 (재귀)
    for child in company[node]:
        showTree(child, level + 1)

print('회사 조직도')
showTree('CEO', 0)