"""
pandas의 DataFrame의 자료를 원격 DB의 테이블에 저장.

[권장]
.env 파일
DB_USER=root
DB_PASS=123

from dotenv import load_dotenv
load_dotenv()

engine = create_engine(
    f"mysql+pymysql://{os.getenv('DB_USER')}:\
        {os.getenv('DB_PASS')}@127.0.0.1:3306/test?charset=utf8mb4")

"""
import pandas as pd
from sqlalchemy import create_engine
import pymysql

data = {
    "code":[10, 11, 12],
    "sang":['환타','데자와','나랑드'],
    "su":[20, 22, 5],
    "dan":[1700, 1800, '1600']
}
try:
    frame = pd.DataFrame(data)
    print(frame)

    #계정명:비밀번호@url number:port number/사용할 db?charset=utf8
    engine = create_engine("mysql+pymysql://root:123@127.0.0.1:3306/test?charset=utf8")
    
    #저장
    frame.to_sql(name='sangdata', con=engine, if_exists='append', index=False)

    #읽기
    df = pd.read_sql('select * from sangdata', engine)
    print(df)

except Exception as err:
    print('err : ', err)




