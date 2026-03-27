"""
matplotlib : 플로팅 라이브러리 , 그래프 생성을 위한 다양한 함수를 제공
matplotlib 사이트 : https://matplotlib.org/
참고 : https://cafe.daum.net/flowlife/SBU0/11
matplotlib + seaborn을 같이 사용해야 시각화가 좋아짐

시각화의 중요성 (정리) : https://brunch.co.kr/@dimension-value/56

chart 용어 정리 (정리)
    Figure : 모든 차트가 그려지는 전체 영역 
        2차원일때 하나의 Figure에 Axis 두개를 가지고 있다
    Axes(plot) : Figure안에 차트가 그려질 수 있는 영역 
    Axis x/y : x축 y축
    tick : 축 위에 눈금
    lable : 축,차트 의 이름 
"""
import numpy as np
import matplotlib.pyplot as plt
# 한글 깨짐 방지
plt.rc('font', family='malgun gothic') # 맥은 Apple gothic
# 한글 깨짐 방지하면 음수가 깨짐. 음수 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

#  line plot
x = ['서울','인천','수원'] # list  좌표== [0, 1, 2]
# x = ('서울','인천','수원') # tuple
# x = {'서울','인천','수원'} # TypeError: unhashable type: 'set' 순서X-> 좌표가 X
y = [5, 3, 7]

# 틱값 조정
plt.xlim([-1, 3])
plt.ylim([0, 10])

# tick 설정: y축의 라벨을 인위적으로 표시할 수 있다
plt.yticks(list(range(0, 10, 3)))

plt.plot(x, y) 
plt.show() # 렌더링해야 표가 나와 # colab을 사용하면 사용하지 X

data = np.arange(1, 11, 2)
plt.plot(data)  # y축으로 들어감, x축 구간은 자동으로 설정 됨.
x = [0,1,2,3,4]
for a, b in zip(x, data):
    plt.text(a, b, str(b))

plt.show()

x = np.arange(10)
y = np.sin(x)
print(x, y)
# plt.plot(x, y)

# 옵션 주기 - 스타일 지정
# plt.plot(x, y, 'bo') # 파란 점
plt.plot(x, y, 'go--', linewidth=2, markersize=12)
plt.show()

# hold : 복수의 plot이 여러개의 차트를 겹쳐 그림
# sin, cos그리기
x = np.arange(0, np.pi * 3, 0.1)
print(x)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.figure(figsize=(10, 5)) # 그래프 전체 크기 조절 너비(w)와 높이(h)를 줌.
plt.plot(x, y_sin, 'r')     # 선그래프
plt.scatter(x, y_cos)  # 산점도
plt.xlabel('x 축')
plt.ylabel('y 축')
plt.title("sine & cosine")
plt.legend(['sine','cosine']) # legend : 범례 - 위치는 정할 수 있다. 자동은 오른쪽
plt.show()
print()

# sub plot : 하나의 Figure를 여러개의 Axes(plot)으로 나누기
plt.subplot(2, 1, 1) # (2행, 1열, 1행)
plt.plot(x, y_sin)
plt.title('sine')
plt.subplot(2, 1, 2) # (2행, 1열, 2행)
plt.plot(x, y_cos)
plt.title('cosine')
plt.show()
print()

# 꺽은선 그래프
irum = ['a', 'b', 'c', 'd', 'e']
kor = [80, 50, 70, 70, 90]
eng = [60, 70, 80, 90, 100]
plt.plot(irum, kor, 'ro-') # 기준값은 x축 , 기준값의 변하는 값은 y축
plt.plot(irum, eng, 'bo--')
plt.ylim([50, 100])
plt.title('시험 점수')
plt.legend(['국어', '영어'], loc=4 ) #loc='best' 는 적당한 자리를 알아서 잡음.
''' 2   1 <-- legend위치   
    3   4'''
plt.grid(True)
fig = plt.gcf() # 이미지로 저장준비 하겠다.
plt.show()
fig.savefig('plot1.png') # 파일로 저장.


# 이미지 읽기
from matplotlib.pyplot import imread
img = imread("plot1.png")
plt.imshow(img)
plt.show()
"""
이미지 falsk에서 추출할때 이미지를 읽어오는 방법
    (정적) 서버에서 저장(static)해서 img src로 읽는방법
    (동적)모듈 사용 chart.js나, JS 라이브러리 사용
"""