'''
이원카이제곱(분포 비율 차이 검정)
동질성 검정 - 두 집단의 분포가 동일한가? 다른 분포인가? 를 검증하는 방법.        
두 집단 이상에서 각 범주(집단) 간의 비율이 서로동일한가를 검정하게 된다. 
두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출된 것인지 검정하는 방법.
동질성 검정실습1) 교육방법(독립변수 x)에 따른 교육생들의 만족도(종속변수 y) 분석 
동질성 검정 survey_method.csv

귀무 가설 : 교육방법에 따른 교육생들의 만족도에 차이가 있다
대립 가설 : 교육방법에 따른 교육생들의 만족도에 차이가 없다
'''
import pandas as pd
import scipy.stats as stats

print('-'*20,'동질성 검정실습1)','-'*20)
# 교육방법에 따른 만족도에 대한 설문조사 수집 결과 자료
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/survey_method.csv")
print(data.head(2), 
    data['method'].unique(), data['survey'].unique()) # [1 2 3] [1 2 3 4 5]

ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.index = ['방법1','방법2','방법3']
ctab.columns=['매우만족','만족','보통','불만족','매우불만족']
print(ctab) # (건수가 아니라)실험에 의해 관측된 분포 비율이라고 보자 - 동질성 검정

# 카이제곱검정
chi2, p , dof , expected = stats.chi2_contingency(ctab)
print(f'p={p}')         # p = 0.5864574374550608
print(f'chi2={chi2}')   # chi2 = 6.544667820529891
print(f'dof={dof}')     # dof = 8 # 임계값 15.51
print(f'예측된 기대비율(expected)\n{expected}')
print('''
        유의확율0.05 < p-value 0.586이므로 귀무가설 채택.
        임계값 15.51, 카이검정통계량 6.544 이므로 채택역에 존재
        우연히 발생한 자료라고 할 수 있다.
        ''')

'''
동질성 검정 실습2) 연령대별(x) sns 이용률(y)의 동질성 검정
20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 
이용 현황을 조사한 자료를 바탕으로 연령대별로 홍보 전략을 세우고자 한다.
연령대별로 이용 현황이 서로 동일한지 검정해 보도록 하자.

귀무가설 : 연령대별로 SNS 서비스별 이용률 현황은 동일(동질)하다 
대립가설 : 연령대별로 SNS 서비스별 이용률 현황은 동일(동질)하지 않다.
'''
print('-'*20,'동질성 검정실습 2)','-'*20)
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/snsbyage.csv")
print(data.head(3))
print(data['age'].unique(), data['service'].unique()) 
# [1:20대, 2:30대, 3:40대] ['F' 'T' 'K' 'C' 'E']

ctab2 = pd.crosstab(index=data['age'], columns=data['service'])
print(ctab)

chi2, p , dof , expected = stats.chi2_contingency(ctab2)
print(f'p={p}')         # p = 1.1679064204212775e-18
print(f'chi2={chi2}')   # chi2 = 102.75202494484225
print(f'dof={dof}')     # dof = 8  # 임계값 15.51
print(f'예측된 기대비율(expected)\n{expected}')
print('''
        유의확율0.05 > p-value 1.1679064204212775e-18이므로 귀무가설 기각.
        임계값 15.51, 카이검정통계량 102.75 이므로 기각역에 존재
        우연히 발생한 자료라고 할 수 없다.
        연령대별로 SNS 서비스별 이용률 현황은 동일(동질)하지 않다라는 의견 인정
        ''')

# 샘플링하기
print("전체건수 : ", len(data)) # 1439
print('위 자료는 샘플 자료이겠으나 모집단이라 가정하고 샘플링 후 검정하기')
samp_data =data.sample(n=500, replace=True, random_state=1)  
# replace=True 복원추출 하겠다., random_state=1 랜덤한값 고정
print(samp_data.head(), ' ', len(samp_data))

# 샘플링 데이터 교차테이블 생성
ctab3 = pd.crosstab(index=samp_data['age'], columns=samp_data['service'])
print(ctab)

# 샘플링 데이터 카이제곱검정하기
chi2, p , dof , expected = stats.chi2_contingency(ctab3)
print(f'p={p}')         # p = 1.6088950709449814e-05
print(f'chi2={chi2}')   # chi2 = 36.20748437710444
print(f'dof={dof}')     # dof = 8  # 임계값 15.51
print(f'예측된 기대비율(expected)\n{expected}')
print('sampling 데이터역시 귀무가설 기각')