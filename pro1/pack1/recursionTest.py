'''
재귀문제 :  리스트 자료 v = [7, 9, 15, 43, 32, 21] 에서 최대값 구하기 - 재귀 호출 사용 

                print(find_max(v, len(v)))
'''

v = [7, 9, 15, 43, 32, 21]
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
# print(v, len(v), max(v))

def find_max(v, line):

        len = line-1
        liv = v

        # 리스트 최대값 인덱스로 비교
        if liv[len] == max(liv):
                print(liv[len])

        # 인덱스값 0보다 작으면 return
        elif len < 0: 
                return print('done')
        
        # 최대값 찾을때 까지 재귀호출
        return find_max(v, len)         



print(find_max(v,len(v)))
print(find_max(a,len(a)))

print('------------'*10)
# 재귀문제 :  리스트 자료 v = [7, 9, 15, 43, 32, 21] 에서 최대값 구하기 - 재귀 호출 사용 
                #   print(find_max(v, len(v)))
# for문으로 
def find_max_for(v):
        max_value = v[0]  

        for i in range(1, len(v)):  
                if v[i] > max_value: 
                        max_value = v[i] 

        return max_value 

v = [7, 9, 15, 43, 32, 21]
print(find_max_for(v))

print('------------'*10)
def find_max(v, n):

# 종료 조건 : 리스트의 첫 번째 값을 반환하고 재귀 종료
        if n == 1: 
                return v[0] 
# 재귀 호출
        prev_max = find_max(v, n - 1)
        # 앞의 (n-1)개 원소 중 최대값을 구함. 이 호출이 끝나야 아래 코드로 내려옴

# 마지막 값과 비교
        if v[n - 1] > prev_max:
                # 현재 단계에서 마지막 원소 v[n-1]과 이전 단계에서 구한 최대값(prev_max)을 비교
                return v[n - 1]  # 마지막 값이 더 크면 그 값을 최대값으로 반환
        else:
                return prev_max 
        

'''
[재귀 함수 순서 : 호출 -> 반환]
호출 ->
f_m(v,5)
        f_m(v,4)
                f_m(v,3)
                        f_m(v,2)
                                f_m(v,1)
                                <-반환
                                v[1] = 9
                        15
                43
        43
43
                                        
'''

v = [7, 9, 15, 43, 32, 21] 
print(find_max(v, len(v)))

print('-- 좀 더 파이썬 스럽게 ---')
def find_max(v, n):
        if n == 1:
                return v[0]    

        return max(
                v[n - 1],           #  현재 단계의 마지막 원소
                find_max(v, n - 1)  #  앞의 (n-1)개 중 최대값을 재귀로 구함
                )   # 두 값 중 큰 값을 반환

print(find_max(v, len(v)))