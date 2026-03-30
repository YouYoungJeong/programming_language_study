from pathlib import Path
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import koreanize_matplotlib

# 순서를 위해 상수지정
BRAND_ORDER = ['스타벅스', '폴바셋','이디야','탐앤탐스']

# 카이제곱 검정하기
def analysiy_func(rdata:list[dict]):
    df = pd.DataFrame(rdata)

    if df.empty:
        return pd.DataFrame(), "데이터가 없어요", pd.DataFrame()
    
    df = df.dropna(subset=['gender','co_survey'])

    # 성별에 따른 커피브랜드 선호 빈도수
    crossTab = pd.crosstab(index=df['gender'], columns=df['co_survey'])

    if crossTab.size==0 or crossTab.shape[0] < 2 or crossTab.shape[1] < 2:
        # 데이터가 없는 경우들
        return crossTab, '표본 자료가 부족해 카이제곱검정 수행 불가', df
    
    # 유의수준 : 0.05(5%)
    alpha = 0.05

    # 이원 카이제곱검정
    chi, p, dof, expected = stats.chi2_contingency(crossTab)
    
    # 데이터 값이 작으면 기대빈도값이 작게 나옴, 
    min_expected = expected.min()
    note = ''
    if min_expected < 5:
        note = f'<br><small>* 주의 : 기대빈도 최소값 {min_expected:.2f} (5 미만)*</small>'

    if p >= alpha:
        results = f"p값{p:.5f} >= {alpha} 이므로 성별에 따라 커피 선호브랜드는 <b>차이가 없다(귀무가설)</b> {note} "
    else:
        results = f"p값{p:.5f} < {alpha} 이므로 성별에 따라 커피 선호브랜드는 <b>차이가 있다(대립가설)</b> {note} "
    
    return crossTab, results, df

# 차트 저장하기
def save_barchart_func(df:pd.DataFrame, out_path:Path) -> bool: 
    # 가독성을 위한 타입 힌트:pd.DataFrame, bool가독 유무확인

    # df가 없거나, 비어있거나 co_survey라는 칼럼이 없으면 False를 반환
    if df is None or df.empty or "co_survey" not in df.columns:
        return False
    
    count = df['co_survey'].value_counts().reindex(BRAND_ORDER, fill_value=0)
    # dir가 없으면
    out_path.parent.mkdir(parents=True, exist_ok=True) # 하위폴더도 만들고, 있으면 스킵

    fig = plt.figure()
    ax = count.plot(kind='bar', width=0.6, edgecolor='black')
    ax.set_xlabel("커피 브랜드명")
    ax.set_ylabel("선호 건수")
    ax.set_title("커피 브랜드별 선호 건수")
    ax.set_xticklabels(BRAND_ORDER, rotation=0)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches='tight') # Path타입 문자열 변환
    plt.close(fig)

    return True