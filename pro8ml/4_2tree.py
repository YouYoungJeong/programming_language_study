'''
의사결정나무 (Decision Tree) 분류 모델
    키, 머리카락 길이데이터로 남여 분류
'''
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np

x = [[180, 15],[177, 42],[156, 35],[174, 65],[161, 25],[160, 45],[170, 65],[155, 55]]
y = ['man','woman','woman','man','woman','man','man','man']
feature_names = ['height','hair_length']
calss_names = ['man','woman']

# 모델 생성하기
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
model.fit(x, y)

# 분류모델 성능 점수
print(f'정확도 : {model.score(x, y)}')
print(f'예측값 : {model.predict(x)}')
print(f'실제값 : {y}')
print()

# 새로운 데이터
new_data = [[177, 78]]
print('new_data pred :', model.predict(new_data))

# 의사결정나무 모델 시각화
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=feature_names, 
        class_names=model.classes_,   # model.class_ 자동으로 class이름을 적어줘
        filled=True, rounded=True, fontsize=12) 
plt.title('의사결정나무 모델 시각화')
plt.show()