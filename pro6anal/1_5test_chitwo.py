"""
이원카이제곱 - 교차분할표(crosstable) 이용
: 두 개 이상의 변인(집단 또는 범주)을 대상으로 검정을 수행한다.
분석대상의 집단 수에 의해서 독립성 검정과 동질성 검정으로 나뉜다.
    독립성 : 독립성 검정은 두 변수 사이의 연관성을 검정
            ex) 부모의 학력수준과 자녀의 대학진학여부 - 둘 사이에 서로 "관련이 있다 없다"의 여부
            ex) 교육 수준과 흡연율이 "관련이 있다 없다"의 여부

    동질성 : 분포 비율의 차이 
실습 : 교육수준과 흡연율 간의 관련성 분석 : smoke.csv'

독립변수(x) : 교육수준
종속변수(y) : 흡연율

귀무가설 : 교육수준과 흡연율 간의 관계가 없다.(독립 이다, 연관성이 없다.)
대립가설 : 교육수준과 흡연율 간의 관계가 있다.(독립이 아니다, 연관성이 있다.)
"""
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/smoke.csv")
print(data.head(3))
print(data.tail(3))
print(data['education'].unique())   # [1:대학원졸 2:대졸 3:고졸]
print(data['smoking'].unique())     # [1:과흡연   2:보통 3:노담]

# 학력 수준별 흡연 빈도수 : 교차표(crosstable)
# 샘플링한 데이터의 차이가 좀 보이네~

'''
# 비율로 출력 - 빈도수 출력이기 때문에 확인만해
ctab = pd.crosstab(index=data['education'], columns=data['smoking'], normalize=True) 
ctab.index = ['대학원졸', '대졸', '고졸']
ctab.columns = ['과흡연', '보통', '노담']
print(ctab)
'''

# 빈도수 출력
ctab = pd.crosstab(index=data['education'], columns=data['smoking'])
ctab.index = ['대학원졸', '대졸', '고졸']
ctab.columns = ['과흡연', '보통', '노담']
print(ctab)

# 이원카이제곱검정
# 방법1. - 일부만 참여할 때는 list값을 추출해서 계산
chi_result = [ctab.loc['대학원졸'],ctab.loc['대졸'],ctab.loc['고졸']]
chi2 , p, dof, expected = stats.chi2_contingency(chi_result)

# 방법2. 데이터값 모두가 참여하기 때문에 crosstab값을 넣어도 됨
# chi2 , p, dof, expected = stats.chi2_contingency(ctab)
# print(chi_result)
print(f'chi2={chi2}')   # 18.910915739853955
print(f'p={p}')         # 0.0008182572832162924
print(f'dof={dof}')     # 4
print(f'예측된 기대도수(expected)\n{expected}')

print('''
    판정1 : 유의수준0.05 > p-value0.00081이므로 귀무가설을 기각.
        교육수준과 흡연율은 관계가 있다. 
        smoke.csv(수집자료)는 우연히 발생된 자료가 아니다.
    ''')

print('''
    판정2 :chi2 = 18.910915, df=4. criticl_value(임계값): 9.49 이므로
        chi2값이 임계값을 넘어가 기각역내에 존재하므로 귀무가설을 기각(임계치 오른쪽 위치) 대립가설 채택\n
    ''')

'''
이후 다양한 자료, 의견 등으로 보고서를 작성, 가설검정은 하나의 부품일 뿐이다. 
흡연율에 대한 보고서일 경우 다양한 경우의 내용중 
현재는 학력별로 확인하고 보고서를 만들어 어느 한 영역을 차지하고 있는것
그래프 비교. 판정, 추후예정까지 보고해야함.
'''

print('-'*20,'독립성 검정 실습 2','-'*20)
'''
남성과 여성의 스포츠 음료 선호도 검정
남여-독립변수(x) : 범주형, 음료선호-종속변수(y) : 범주형
귀무가설(H₀) : 성별과 음료 선호는 서로 관련 없다.
대립가설(H₁) : 성별과 음료 선호는 서로 관련 있다.
'''
data = pd.DataFrame({ # 교차표 생성
    '게토레이':[30, 20],
    '포카리':[20, 30],
    '비타500':[10, 30]
}, index=['남성','여성'])
print(data)
chi2 , p, dof, expected = stats.chi2_contingency(data)
print(f'p={p}')         # p = 0.003388052521834713 
print(f'chi2={chi2}')   # chi2 = 11.375
print(f'dof={dof}')     # dof = 2
print(f'예측된 기대도수(expected)\n{expected}')
'''

관측값 게토레이  포카리  비타500
남성    30   20     10
여성    20   30     30

예측된 기대도수(expected)
[[21.42857143 21.42857143 17.14285714]
[28.57142857 28.57142857 22.85714286]]

서로 관련이 있느냐 없는냐에 대한 판정을 해야함
'''
print('''
    판정1 : 예상된 기대도수와 관측값은 서로 관련이 있는가?
        유의수준0.05 > p = 0.0033 이므로 귀무가설을 기각
        따라서 성별과 음료 선호는 서로 관련 있다.
    ''')

# heatmap시각화
sns.heatmap(data=data, annot=True, fmt='d', cmap='Blues')
#annot=True 셀안에 숫자표시
plt.title('성별에 따른 음료 선호')
plt.xlabel('음료')
plt.ylabel('성별')
plt.show()