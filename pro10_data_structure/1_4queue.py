'''
큐(Queue)
    선입선출(First In, First Out, FIFO) 원칙을 따른다. 

    List 대신 deque를 이용해서 Queue를 구현함.

deque의 주요 메소드
    deque(),
    append(1)       : 우측에 추가
    appendleft(1)   : 좌측에 추가
    pop()           : 우측 제거
    popleft()       : 좌측 제거
    ...
'''
from collections import deque

# 놀이공원 대기 줄
queue = deque()
print("놀이 공원 기구 대기 시작")

# 줄서기
queue.append('철수') # 우측에 추가
print('첫번째 줄서기 :', list(queue)) # deque(['철수'])에서 deque를 없애려면 list로 출력
queue.append('영희') # 우측에 추가
print('두번째 줄서기 :', list(queue))
queue.append('민수') # 우측에 추가
print('세번째 줄서기 :', list(queue))

print()
# 놀이 기구 탑승 - FIFO (중간데이터 접근 불가)
first_person = queue.popleft() # 좌측 사람이 queue에서 제거
print(first_person, '놀이기구 탑승')
print('현재 대기줄 :', list(queue))
print()

# 한명더 놀이기구 탑승
first_person = queue.popleft() # 좌측 사람이 queue에서 제거
print(first_person, '놀이기구 탑승')
print('현재 대기줄 :', list(queue))
print()

if queue:
    print('탑승 예정자 :', queue[0])
else:
    print("대기자 없음")

# ==============================================================================
# FIFO를 class로 연습
# ==============================================================================
print('-------'*10)
class MyQueue:
    def __init__(self, iterable = None):
        self._data = deque() # deque를 내부저장소로 사용
        if iterable is not None:
            for x in iterable:
                self.enqueue(x)
    
    # Rear(back)에 요소 추가(enqueue) 메소드
    def enqueue(self, x):
        self._data.append(x) 
        return x
    
    # Front(앞) 요소 제거(dequeue) 메소드
    def dequeue(self):  
        if not self._data: # 맨위가 비어있다면
            raise IndexError("Queue가 비어 있습니다.")
        return self._data.popleft()
    
    # 조회(확인)만
    # Queue에서 맨앞(Front)요소를 확인하는 메소드
    def front(self):
        if not self._data: # 맨위가 비어있다면
            raise IndexError("Queue가 비어 있습니다.")
        return self._data[0] 
    
    # 비어있는지 안비어있는지 확인하는 메소드
    def is_empty(self):
        return not self._data # 비었을 때 True 반환

    # 요소의 갯수 반환하는 메소드
    def size(self):
        return len(self._data)
    
    # Queue를 비우는 메소드
    def clear(self):
        self._data.clear()

    # 출력(Front -> Rear(back)순으로 출력되는 특별 메소드)
    # Python 실행시 자동 호출(print) 되는 특별 메소드
    def __repr__(self):
        return f'Queue(Front -> Rear(back) {list(self._data)})'

def demo1Func():
    imsi1 = MyQueue()
    imsi2 = MyQueue([10, 20, 30])
    print(imsi1)
    print(imsi2)
    print(imsi2.front())
    print(imsi2.size())
    imsi2.clear()
    print(imsi2)
    print("----------------")
    
    q = MyQueue()
    for item in ['A','B','C','D']:
        q.enqueue(item)
        print(f'enqueue {item} -> {q}')
    
    print('\nFIFO 하나씩 추출')
    while not q.is_empty():
        print(f'dequeue -> {q.dequeue()} | Now : {q}')

def demoFunc2(jobs, ppm=15):
    q = MyQueue(jobs)   # 작업들 q에 입력
    t_sec = 0.0         # 시뮬레이션 시간 누적(꾸밈)
    order = []          # 실제 처리된 문서를 저장하는 List

    print("프린터로 출력하기")
    while not q.is_empty():
        doc, pages = q.dequeue() # 하나씩 jobs 추출
        # 출력시간 계산하기 : (페이지수 / 분당 페이지수) * 60
        duration = (pages / ppm) * 60.0
        t_sec += duration
        order.append(doc) # 처리 순서 기록
        print(f"t={t_sec:6.1f}초 | 출력 : {doc:10s}({pages}페이지)")
    print(f'처리 순서(FIFO) : {order}')

if __name__ == "__main__":
    demo1Func()
    print('문서 프린터로 출력 시뮬레이션 - FIFO')
    jobs = [('abc.pdf', 10), ('nice.doc', 30), ('good.txt', 5),] # 문서이름
    demoFunc2(jobs, ppm=20) #page/min = 20 -> 1분에 20장 출력 : 페이지수