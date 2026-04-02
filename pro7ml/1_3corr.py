'''
상관관계 분석(스토리가 있는)
국가별 외국인의 한국 방문 출입국 데이터
https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=fe6b1ac0-6271-11ea-8b67-7b32ce18203a

정제 : 외국인(人:미, 일, 중), 국내 관광지(5개)
외국인(人:미, 일, 중)이 국내 관광지(5개) 방문 관련자료 사용
나라별 관광지 상과관계 확인하기
'''
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 수치표기, dataframe 전체보기
np.set_printoptions(suppress=True, precision=10)
pd.options.display.float_format = '{:.10f}'.format
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# 산점도 그리기
def setScatterGraph(tour_table, all_table, tourpoint):
    # print(tourpoint) # 잘들어오는지 확인 ok
    
    # 계산할 관광지명에 해당하는 자료만 뽑아 tour에 저장하고 외국인 자료와 병합
    tour = tour_table[tour_table['resNm'] == tourpoint]
    # print('tour : ',tour)
    '''
    tour :         resNm  ForNum
            yyyymm
            201101   창덕궁   14137
    '''              
    
    # 서울시 관광지 정보파일과 세 외국인 테이블 합치기                                                                                                                   
    merge_table = pd.merge(tour, all_table, left_index=True, right_index=True)
    # print("merge_table : ", merge_table)
    '''
    merge_table : resNm  ForNum   china   japan    usa
        yyyymm
        201101   창덕궁   14137   91252  209184  43065
    '''

    # 시각화 -  상관계수 계산
    fig = plt.figure()
    fig.suptitle(tourpoint + " 상관관계 분석")
    # 나라가 3개니까 1행 3열로 출력하겠다(산점도- scatter)
    plt.subplot(1, 3, 1)
    plt.xlabel('중국인 방문객 수')
    plt.ylabel('외국인 방문객 수')
    # 상관계수 계산하기 - 람다객체생성
    lamb1 = lambda p:merge_table['china'].corr(merge_table['ForNum'])
    r1 = lamb1(merge_table)
    # print('r1 :',r1)
    plt.title('r = {:.5f}'.format(r1))
    plt.scatter(merge_table['china'], merge_table['ForNum'], alpha=0.7, s=6, c='red')

    plt.subplot(1, 3, 2)
    plt.xlabel('일본인 방문객 수')
    plt.ylabel('외국인 방문객 수')
    # 상관계수 계산하기 - 람다객체생성
    lamb2 = lambda p:merge_table['japan'].corr(merge_table['ForNum'])
    r2 = lamb2(merge_table)
    # print('r2 :',r2)
    plt.title('r2 = {:.5f}'.format(r2))
    plt.scatter(merge_table['japan'], merge_table['ForNum'], alpha=0.7, s=6, c='green')
    
    plt.subplot(1, 3, 3)
    plt.xlabel('미국인 방문객 수')
    plt.ylabel('외국인 방문객 수')
    # 상관계수 계산하기 - 람다객체생성
    lamb3 = lambda p:merge_table['usa'].corr(merge_table['ForNum'])
    r3 = lamb3(merge_table)
    # print('r3 :',r3)
    plt.title('r3 = {:.5f}'.format(r3))
    plt.scatter(merge_table['usa'], merge_table['ForNum'], alpha=0.7, s=6, c='blue')
    plt.tight_layout()    
    plt.show()
    
    return [tourpoint, r1, r2, r3] # 광광지명, 상관계수 들고 r_list가 받아


def processFunc():
    # 서울시 관광지 정보 파일
    fname = "corr3_json/서울특별시_관광지입장정보_2011_2016.json"
    jsonTP = json.loads(open(fname, 'r', encoding='utf-8').read())
    tour_table = pd.DataFrame(jsonTP, columns=('yyyymm','resNm','ForNum'))# 년월, 관광지명, 입장객수
    tour_table = tour_table.set_index('yyyymm')
    # print(tour_table)
    '''
                resNm  ForNum
    yyyymm
    201101     창덕궁   14137
    201101     운현궁       0
    '''
    resNm = tour_table.resNm.unique()
    # print("resNm(5개만) :",resNm[:5]) # 앞에 5개만 추출 ['창덕궁', '운현궁', '경복궁', '창경궁', '종묘']


    # 중국인 관광객 정보 파일 DataFrame에 저장
    cdf = "corr3_json/중국인방문객.json"
    jdata = json.loads(open(cdf,'r', encoding='utf-8').read())
    china_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))
    china_table = china_table.rename(columns={'visit_cnt':'china'}) # 중국인 방문객수 col이름 변경
    china_table = china_table.set_index('yyyymm')
    # print(china_table[:2])

    # 일본인 관광객 정보 파일 DataFrame에 저장
    jdf = "corr3_json/일본인방문객.json"
    jdata = json.loads(open(jdf,'r', encoding='utf-8').read())
    japan_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))
    japan_table = japan_table.rename(columns={'visit_cnt':'japan'}) # 일본인 방문객수 col이름 변경
    japan_table = japan_table.set_index('yyyymm')
    # print(japan_table[:2])

    # 미국인 관광객 정보 파일 DataFrame에 저장
    udf = "corr3_json/미국인방문객.json"
    jdata = json.loads(open(udf,'r', encoding='utf-8').read())
    usa_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))
    usa_table = usa_table.rename(columns={'visit_cnt':'usa'}) # 미국인 방문객수 col이름 변경
    usa_table = usa_table.set_index('yyyymm')
    # print(usa_table[:2])

    # 세나라 데이터 합치기
    all_table = pd.merge(china_table, japan_table, left_index=True, right_index=True)
    all_table = pd.merge(all_table, usa_table, left_index=True, right_index=True)
    # print(all_table) # [72 rows x 3 columns]

    r_list = [] # 세나라 상관변수 담을 리스트
    for tourpoint in resNm[:5]:
        r_list.append(setScatterGraph(tour_table, all_table, tourpoint))

    # print(r_list)
    r_df = pd.DataFrame(r_list, columns=('고궁명','중국','일본','미국'))
    r_df = r_df.set_index('고궁명')
    # print(r_df)
    '''
                중국            일본            미국
    고궁명
    창덕궁 -0.0587911041  0.2774443570  0.4028160633
    운현궁  0.4459448838  0.3026152183  0.2812576500
    경복궁  0.5256734294 -0.4352281861  0.4251372639
    창경궁  0.4512325398 -0.1645858940  0.6245403780
    종묘   -0.5834218987  0.5298702802 -0.1211266683
    '''

    # 상관관계 그래프그리기
    r_df.plot(kind='bar', rot=50)
    plt.show()






if __name__ == "__main__":
    processFunc()