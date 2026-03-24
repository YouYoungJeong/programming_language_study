"""
seaborn 라이브러리 : matplotlib의 기능 보충용

** 시각화 한글깨짐 방지 **
koreanize-matplotlib 를 사용 - colab에서도 유효!
pip install koreanize-matplotlib
import koreanize_matplotlib
"""
import matplotlib.pyplot as plt
import koreanize_matplotlib # 한글 깨짐 방지.
import seaborn as sns
import pandas as pd

# seaborn이 타이타닉 데이터를 제공하고 있음
titanic = sns.load_dataset('titanic')
print(titanic.info(max_cols=None)) # info정보 짤리는경우 max_cols=None 사용
sns.displot(titanic['age'])
plt.title('나이 차트')
plt.show()

# seaborn box plot
sns.boxplot(y='age', data=titanic, palette='Paired')
plt.show()

# seaborn 산점도
sns.relplot(x='sex', y='age', data=titanic)
plt.show()

# seaborn heatmap 피벗테이블의 결과 보기
# 클래스별 성별 건수
# 누적건수 이기 때문에 밀도가 높을 수록 색이 진해짐.
titanic_pivot = titanic.pivot_table(index='class', columns='sex', aggfunc='size')
print(titanic_pivot)
sns.heatmap(titanic_pivot, cmap=sns.light_palette("gray"), annot=True, fmt='d')
plt.show()

"""
Boxplot 기준 이상치(outlier)의 정의
"""
# 1. 데이터 정의
data = [10, 12, 13, 15, 14, 12, 11, 100]
df = pd.DataFrame({'score': data})

# 2. IQR 기반 이상치 탐지
Q1 = df['score'].quantile(0.25)
Q3 = df['score'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 3. 이상치, 정상치 분리
outliers = df[(df['score'] < lower_bound) | (df['score'] > upper_bound)]
filtered_df = df[(df['score'] >= lower_bound) & (df['score'] <= upper_bound)]

# 4. 이상치 출력
print("이상치 값:")
print(outliers)

# 5. 박스플롯 시각화: 제거 전/후 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 이상치 포함
sns.boxplot(y=df['score'], ax=axes[0], color='salmon')
axes[0].set_title('이상치 포함 데이터')
axes[0].set_ylabel('Score')
axes[0].grid(True)

# 이상치 제거 후
sns.boxplot(y=filtered_df['score'], ax=axes[1], color='lightblue')
axes[1].set_title('이상치 제거 후')
axes[1].set_ylabel('Score')
axes[1].grid(True)

plt.tight_layout()
plt.show()