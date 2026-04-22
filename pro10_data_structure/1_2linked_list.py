'''
연결된 리스트 (Linked List)
    임의의 공간에 기억시키고, 순서에 따라 포인터로 자료를 연결시킨 구조
    검색은 느리지만 삽입 삭제는 빠르다.
        예시) 놀이공원 줄서기
    
    종류
        단순 연결 리스트(Singly Linked List)
        단순 원형 연결 리스트 (Singly Circular Linked List)
        이중 연결 리스트(Doubly Linked List)
        이중 원형 연결 리스트(Doubly Circular Linked List)

Linear / Linked 특징 비교
항목       | Linear List        |   Linked List
--------------------------------------------------------
저장 방식  | 연속된 공간        | 흩어진 공간
접근       | 빠름 (index)       |  느림 (처음부터 따라감)
삽입/삭제  | 느림 (이동 발생)   | 빠름 (연결만 바꿈)

'''

# 하나의 노드를 가지고 있는 class 생성 - 노드 객체 생성
class Node:
    # 생성자(init)
    def __init__(self, name): 
        # 이름과 다음 사람을 가르기는 next주소를 가짐
        self.name = name    # 입력 받은 사람
        self.next = None    # 다음 사람(pointer)

# 연결 리스트를 관리하는 class 생성
class LinkedList:
    # head 생성
    def __init__(self):
        self.head = None # 맨 앞사람의 주소(리스트의 시작점, head)
    
    # 새로운 Node추가(줄 뒤에 다음사람 추가)
    def append(self, name):
        new_node = Node(name) # 새 Node 생성
    
        # 첫번째 Node 즉 List가 비어 있는 경우
        if self.head is None:
            self.head = new_node
            return
        
        # 이미 Node가 있다면 마지막 Node까지 이동(줄의 맨 끝 찾기)
        current = self.head # head부터 시작
        
        while current.next:
            
            # 다음 사람이 있다면 current.next값을 current에 넘겨줌
            current = current.next 
        
        # 마지막 노드를 찾았으면 새로운 노드값 주기
        current.next = new_node

    # 출력하기
    def show(self):
        # 주소 보기
        # print(line.head)
        # print(line.head.next)
        # print(line.head.next.next)
        current = self.head
        while current: # current라는 객체가 있다면(True)
            print(current.name, end=" -> ")
            current = current.next # 포인터 이동
        print('끝')
    
    # 특정 노드 뒤에 새 노드 끼워넣기
    # target Node를 찾기 -> 새 Node만들기 -> 기존 연결 변경
    def insert_after(self, target_name, new_name):
        # 탐색을 위해 head point설정
        current = self.head
        
        # 현재 Node이름이 target Node와 같다면 
        while current:
            if current.name == target_name:
                new_node = Node(new_name) # 새 노드 생성
                
                # 추가된 노드 앞뒤의 노드들이 주소를 가르키는 방향을 바꿔줘야함
                new_node.next = current.next
                current.next = new_node 
                # 삽입이 끝나면 insert_after 함수를 나옴
                return
            # target_name을 만날때 까지 다음 Node로 넘어감
            # target_name을 만나면 만나지 않음
            current = current.next
    
    # 특정 사람 삭제
    def remove(self, name):
        # 첫 Node가 삭제 대상인 경우(head) - 두번째 사람의 Node주소로 head 변경
        if self.head and self.head.name == name:
            self.head = self.head.next
            return
        
        # 첫 Node가 삭제 대상이 아닌경우 찾기
        current = self.head
        while current and current.next: # current와 current.next가 모두 존재하는 경우
            if current.next.name == name:
                current.next = current.next.next # 다음사람 -> 다음다음 사람
                return
            current = current.next


line = LinkedList()
line.append('철수') # head는 첫번째 사람 '철수'를 기억하고 있는 상태
line.append('영희')
line.append('민수')

print('현재 줄 상태 :')
line.show()
print()

# 민수 앞에 지수를 삽입
line.insert_after('영희','지수')
print('지수를 삽입 줄 상태 : ')
line.show()
print()

# 영희가 줄서기를 포기(삭제)
line.remove('영희')
print('영희 삭제 후 줄 상태 :')
line.show()
print()