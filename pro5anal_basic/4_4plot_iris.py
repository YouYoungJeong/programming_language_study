"""
iris dataset사용
150행, 
3가지 종류(Setosa, Versicolor, Virginica), 
4개 특성 (Sepal Length, Sepal Width, Petal Length, Petal Width)

추가 :  https://cafe.daum.net/flowlife/SBU0/57 한번 해봐
"""
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline # 매직 명령어 jupyter notebook에서 실습시 show() 생략
import koreanize_matplotlib

iris_data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/iris.csv")
print(iris_data.info())
print(iris_data.head(3))
print(iris_data.tail(3))

# 산점도 그리기 # 꽃받침, 꽃잎의 길이
plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'])
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.title('붓꽃(iris) data')
plt.show()
print()

# 색 넣어서 보기
# 중복값 제거 ['setosa' 'versicolor' 'virginica']
print(iris_data['Species'].unique()) # 리스트로 나오니까 순서 O
print(set(iris_data['Species'])) # 순서는 X
cols=[] # 꽃의 종류에 따라 다른색 기억하는 공간 생성
for s in iris_data['Species']:
    choice = 0
    if s =='setosa': choice=1
    elif s == 'versicolor': choice=2
    elif s == 'virginica': choice=3
    cols.append(choice)
plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'], c=cols)
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.title('붓꽃(iris) data')
plt.show()
print()

# pandas의 시각화 기능
from pandas.plotting import scatter_matrix
iris_col = iris_data.loc[:,'Sepal.Length':'Petal.Width']
print(iris_col.info()) 
scatter_matrix(iris_col, diagonal='kde') # diagonal 밀도 분포
plt.show()

# Seaborn 사용하기
import seaborn as sns
sns.pairplot(iris_data, hue='Species', height=2)
plt.show()