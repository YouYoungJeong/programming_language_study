'''
상관관계 문제)
https://github.com/pykwon/python 에 있는 Advertising.csv 파일을 읽어 
tv, radio, newspaper 간의 상관관계를 파악하시오. 
또한 sales와 관계를 알기 위해 sales에 상관 관계를 정렬한 후 
TV, radio, newspaper에 대한 영향을 해석하시오.
그리고 이들의 관계를 heatmap 그래프로 표현하시오. 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
pd.set_option('display.max_columns', None)


data = pd.read_csv("Advertising.csv")
data = data.set_index('no')
# print(data) # [200 rows x 5 columns]

#tv, radio, newspaper 간의 상관관계를 파악
print(data.corr())
print()


# sales에 상관 관계를 정렬
data_corr = data.corr()
print(data_corr['sales'].sort_values(ascending=False))
print()
'''
tv          0.7822244249
radio       0.5762225746
newspaper   0.2282990264

-> tv와 sales의 상관관계가 가장 강한 양적 상관관계를 나타낸다.
-> radio와 sales의 상관관계는 뚜렷한 양적 상관관계를 나타낸다.
-> newspaper와 sales의 상관관계는 약한 양적 상관관계를 나타낸다.
-> 결론 : tv광고가 제일 판매량에 영향을 많이 주고 그다음이 radio광고가 영향을 준다.
    newspaper가 판매량과의 상관관계가 가장 낮았다.
'''

# 이들의 관계를 heatmap 그래프로 표현
sns.heatmap(data_corr, annot=True)
plt.show()



