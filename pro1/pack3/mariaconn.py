# config정보를 따로 피클링 저장
import pickle

config = {
    'host':'127.0.0.1',
    'user' : 'root',
    'password':'123',
    'database':'test',
    'port':3306, 
    'charset':'utf8'
}
# mydb.dat로 객체를 바이너리로 저장 하겠다. <= 교육용 이름 통일!
with open('mydb.dat', mode='wb') as obj:
    pickle.dump(config, obj)