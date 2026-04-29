'''
cost function(비용함수)
    비용 함수(Cost Function)는 머신러닝 모델의 예측값(y^)과 실제값(y) 사이의 
    오차를 수치화한 함수로, 학습 목표는 이 비용 함수의 값을 최소화하는 것입니다. 
    모델의 정확도를 측정할 때 활용되며 예측값(y^)과 실제값(y)의 평균을 의미함
    평균 제곱 오차(MSE: Mean Squared Error)
    수식 :
    인공신경망은 델타규칙(경사하강법)으로 W(weight)와 B(bias)를 갱신한다
    경사하강법은 최소제곱법 대신에 평균제곱오차(MSE)를 정의하고, 
    그 오차를 최소화 하기 위해 경사하강법을 반복적으로 사용해 파라미터를 갱신하다.
        전통적인 방법은 : 최소제곱법을 사용

DL은 W와 B를 제공
곡선에 대한 접선의 기울기를 구해
기울기가 + 왼쪽으로 학습, -인경우 오른쪽으로 학습
기울이가 0일 때 cost가 가장 적어
ML인경우 R2, Acc가 좋아진다
DL은 에폭을 크게주고 조기종료 와 학습률(성큼성큼)을 준다.
'''

'''
비용함수 구하기
y - y^의 차이가 작을 때 cost는 0에 근사한다.
wx + b수식에서 w와 b를 최적의 추세선이 만들어지도록 갱신해야 한다. 
'''
import math
import numpy as np

real = np.array([10, 9, 3, 2, 11])      # y(실제값)
# pred = np.array([11, 5, 2, 4, 3])     # y^(예측값) 차이가 큰 경우   cost : 11.6
pred = np.array([10, 8, 3, 4, 10])      # y^(예측값) 차이가 작은 경우 cost : 1.2

cost = 0
for i in range(len(real)):
    cost += math.pow(pred[i] - real[i], 2)
    print(cost)

print('cost :',cost / len(real))
print()

# 최적의 W(weight, 가중치) 얻기의 이해
# 선형회귀 모델 수식은 hypothesis(y^) = w * x + b
# cost = tf.reduce_sum(tf.pow(hypothesis - y), 2)/len(y) -- tensorflow수식
# cost = tf.reduce_mean(tf.square(hypothesis - y)) # 위의 수식과 같다 -- tensorflow수식
print('최적의 W(weight, 가중치) 얻기의 이해')

import tensorflow as tf
import matplotlib.pyplot as plt
import koreanize_matplotlib

x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]
b = 0 # bias는 편의상 0으로 준다

# 시각화를 위한 변수 선언
w_val = []
cost_val = []

for i in range(-30, 50):
    feed_w = i * 0.1 # learning_rate = 0.1
    # print('feed_w :',feed_w) # w의 움직임
    hypothesis = tf.multiply(feed_w, x) + b     #  w * x + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    cost_val.append(cost)
    w_val.append(feed_w)
    print(f'{i} , cost : {cost.numpy()}, weight:{feed_w}')

plt.plot(w_val, cost_val, marker='o')
plt.xlabel("w(가중치, weight)")
plt.ylabel("cost(손실,비용함수)")
plt.show() # 10 , cost : 0.0, weight:1.0