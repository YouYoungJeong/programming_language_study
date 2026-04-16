'''
최근접 이웃(K-Nearest Neighbors)
    최근접 이웃 알고리즘은 우리가 예측하려고 하는 임의의 데이터와 가장 가까운 거리의 데이터 
    K개를 찾아 다수결에 의해 데이터를 예측하는 방법이다
    거리기반 모델이므로 크기가 영향을 미친다 따라서 스케일링이 필요
K값을 잘주는게 제일 중요하다
'''
from sklearn.neighbors import KNeighborsClassifier 
# regressoion은 linear을 더 많이 사용함
import matplotlib.pyplot as plt
import koreanize_matplotlib

train = [
    [5, 3, 2],
    [1, 3, 5],
    [4, 5, 7]
]
label = [0, 1, 1]

plt.plot(train, 'o')
plt.xlim([-1, 5])
plt.ylim([0, 8])
plt.show()

kmodel = KNeighborsClassifier(n_neighbors=3, weights='distance')
kmodel.fit(train, label)
kpred = kmodel.predict(train)
print("pred : ",kpred)
print(f"test acc : {kmodel.score(train, label)}")

new_data = [[1, 2, 9], [6, 2, 1]]
new_pred = kmodel.predict(new_data)
print(f"new_pred : {new_pred}")