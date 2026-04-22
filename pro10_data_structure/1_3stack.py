'''
스택(Stack) : LIFO구조    
    python의List를 Stack처럼 사용
        why? Python에 Stack이 없고 Java에 있지만 현재 잘 안씀
'''
stack = []
print("놀이 공원 입장")

# PUSH
# 놀이 기구 탈 때의 기록을 남김
stack.append('T-express  탑승') 
print('기록 : ', stack)

stack.append("바이킹 탑승")
print('기록 : ', stack)

# Stack에서 주의할점!!
print(stack[1]) # Python의 List의 기능을 사용함. Stack의 기능이 아님!

stack.append("회전목마 탑승")
print('기록 : ', stack)
print()

# POP - 주의!! pop(0), pop(1) 사용X - Stack 개념 위반
# 가장 최근 기록 삭제 
last_action = stack.pop()   
print("마지막 기록 취소(pop())후 현재 :", stack)

last_action = stack.pop()
print("마지막 기록 취소(pop())후 현재 :", stack)
print()

# ==============================================================================
# LIFO를 class로 연습
# ==============================================================================
print('-------'*10)

class MyStack:
    def __init__(self, iterable = None):
        # __data : (-)
        # _data:내부 저장소임을 알려줌(가독성), 기본문법은 아님
        self._data = [] 
        if iterable is not None:
            for x in iterable:
                self.push(x)
    
    # 맨위(top)요소 삽입/추가
    def push(self, x):
        self._data.append(x)
        return x

    # 맨위(top)요소 제거
    def pop(self):
        if not self._data: # 맨위가 비어있다면
            raise IndexError("스택이 비어 있습니다.")
        return self._data.pop()
    
    # 비어있는지 안비어있는지 확인하는 메소드
    def is_empty(self):
        return not self._data # 비었을 때 True 반환

    # Python 실행시 자동 호출(print) 되는 특별 메소드
    def __repr__(self):
        top_to_bottom = list(reversed(self._data))
        return f'Stack(top -> bottom {top_to_bottom})'

def demo1Func():
    print("LIFO에 따라 하나씩 PUSH")    
    s = MyStack()
    for item in ["A","B","C","D"]:
        s.push(item)
        print(f'push {item} -> {s}')
    # print(s._data)
    
    print()
    print("LIFO에 따라 하나씩 꺼내기-POP")
    while not s.is_empty():
        print(f'pop -> {s.pop()}, | 현재는 : {s}')

def demo2Func(text : str) -> str: # 입력 str-> 출력 str : 가독성을 위해 type표시
    s = MyStack(text)
    out = [] # 뒤집힌 문자열 기억용
    while not s.is_empty():
        out.append(s.pop())
    return ''.join(out)



if __name__ == "__main__":
    demo1Func()
    print(demo2Func('Python is good')) # 한글짜씩 차곡차곡 들어가서 뒤집어서 보임.
    print(demo2Func('파이썬 만세')) 