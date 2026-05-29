# pip install gymnasium - 강화학습 환경 사용하기 위해 설치 필요

import math
import numpy as np
# 실습환경 제공하는 Lib (강화 학습)
    # 현재상태가 제공되고 에이전트가 행동을 선택할 수 있다
    # -> 환경이 행동을 반영함 
    # -> 다시 새로운 상태,보상,종료조건 등을 반환
import gymnasium as gym
from gymnasium import spaces # 행동 공간, 관측 공간을 정의 할 수 있다
import matplotlib.pyplot as plt

# 환경(World)/장애물/라이다 설정하기
WORLD_W, WORLD_H = 20.0, 15.0    # 환경 크기 설정
OBSTACLES = [                    # 장애물 (x, y, r)
                (6.0, 4.0, 0.5),    
                (8.0, 10.0, 1.5), 
                (15.0, 5.0, 1.0)
]
NUM_RAYS = 20                   # RAY 수
FOV = np.deg2rad(150)           # 전방 시야각
MAX_RANGE = 8.0                 # 라이다 최대 감지거리(유효 최대 사거리)
STEP_MARCH = 0.5                # RAY 전진 단위 거리

# 좌표값(Ray)이 시뮬레이터 공간 경계 내에 있는지 여부를 판단
def inside_worldFunc(x, y): 
    return (0.0 <= x <= WORLD_W) and (0.0 <= y <= WORLD_H)

# 라이다 광선의 끝점이 장애물과 충돌했는지 여부 -> 원 내부에 있으면 TREU값을 반환
# px, py : 레이의 도달지점
def hit_circleFunc(px, py, cx, cy, r):
    return (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2 # ... > r ** 2 : False값 반환

# 에이전트(행동의 주체 : x, y, θ)에서 시야각(FOV)으로 NUM_RAYS개의 광선을 쏴,
# 각 레이가 처음 부딪히는 지점까지의 거리, 레이 균등 각도 반환값 구하는 함수 생성
def cast_lidar(x, y, theta, num_rays = NUM_RAYS, fov=FOV, max_range = MAX_RANGE, step=STEP_MARCH):
    # 전방 시야각 가장 왼쪽 각도 계산(첫번째 레이 시작 각도)
    start = theta - fov / 2     
    # 균등 분포 각도 배열 계산 (왼쪽 -> 오른쪽)
    angles = start + np.arange(num_rays) * (fov / max(num_rays -1, 1))
    print('angles : ',angles)
    # 초기 거리 배열 초기화 = 최대 거리
    # 각각의 ray마다 장애물을 만났는지 안만났는지를 배열로 확인할 수 있다.
    dists = np.full(num_rays, max_range, dtype=np.float32)

    # 각 ray에 대한 반복 작업
    for i, ang in enumerate(angles):
        dist = 0.0      # 광선(Ray)의 전진값
        hit = False     # 에이전트가 환경(Wrold)영역 밖 또는 장애물 충돌 여부 <- True일때 하나의 에피소드가 끝남.
        
        # 최대 거리까지 전진 -  라이더의 최대 감지 거리값보다 dist가 작을때 까지 반복진행
        while dist < max_range: 
            px = x + math.cos(ang) * dist   # Ray 끝점 x좌표
            py = y + math.sin(ang) * dist   # Ray 끝점 y좌표
            
            # 경계내에 있는지 (공간내 있는지) 파악하는 함수
            if not inside_worldFunc(px, py): 
                hit = True
                break # hit = True 이면 에피소드 끝(while문 빠져나감)
            
            # 장애물 충돌 여부 확인
            for (cx, cy, r) in OBSTACLES:
                if hit_circleFunc(px, py, cx, cy, r):
                    hit = True
                    break # hit = True 이면 에피소드 끝(for문 빠져나감)
            if hit:
                break # hit = True 이면 에피소드 끝(while문 빠져나감)

            dist += step # 충돌이 없으면 0.5만큼 Ray 전진

        dists[i] = min(dist, max_range) # 충돌 거리값 기억하기 없으면 max_range
    
    return dists, angles # 거리 배열, 각도 반환


# Gymnasium 환경 내에서 진행하기 ==================================================
# Gymnasium의 기존환경을 상속받은 클래스 작성하기
class SimpleLiderEnv(gym.Env): # gym.Env을 상속
    def __init__(self, render_mode='human'):
        super().__init__() # 부모클래스 생성자 호출
        self.render_mode = render_mode
        
        # 강화학습 환경 설정시 두가지는 반드시 선언을 해야함
        # 1. 행동 공간
        self.action_space = spaces.Discrete(3) # 가능한 행동 3가지(0:좌회전, 1:직진, 2:우회전)
        # 2. 관측 공간
        self.observation_space = spaces.Box(
            low = 0.0,              # 각 값의 범위는 0.0 
            high = MAX_RANGE,       # ~ 8.0
            shape = (NUM_RAYS,),    # 관측값은 길이 20 짜리 배열, 
            dtype = np.float32      # 자료형은 float
        )

        self.v = 0.25                     # 에이전트의 전진 속도
        self.steer_delta = np.deg2rad(8)  # 에이전트의 회전 각도 - 각 8도를 줌
        self.goal = np.array([18.0, 12.0], dtype=np.float32) # 에이전트의 최종 목표 좌표
        self.goal_radius = 0.6            # 목표 판정을 위한 반경 설정하기 - 점을 기준으로 원을 그려줘야함
        self.max_steps = 400              # 하나의 에피소드에서 허용되는 최대 행동 횟수 400번으로 제한 
        
        self.fig, self.ax = None, None  # 렌더링을 위한 객체 생성
        self._state = None              # private [x, y, θ]
        self._prev_goal_dist = None     # private 이전 목표 거리
        self._steps = 0                 # private step counter

    # Gymnasium의 내장 함수 선언 ----------------------
    # 현재 환경 상태를 강화학습 모델이 이해 할 수 있는 숫자 매열로 만들어 반환하는 함수
    def _get_obs(self):
        x, y, th = self._state # Gymnasium환경에서는 항상 self._satat가 환경 객체가 가진 상태를 가지고 있음
        obs, _ = cast_lidar(x, y, th) # 라이다 거리 관측(앵글은 안받음)
        return obs.astype(np.float32) # 거리를 반환

    # Gymnasium의 환경에 추가 정보를 제공하는 함수
    def _get_info(self):
        x, y, _ = self._state
        d = np.linalg.norm(np.array([x, y]) - self.goal) # 목표까지의 거리
        return {'goal_dist':float(d), 'steps':self._steps}
    
    # 충돌과 관련된 함수
    def _collision(self):
        x, y, _ = self._state
        # 경계 바깥에 있다고 하면 충돌이라 판단하겠다
        if not inside_worldFunc(x, y):
            return True
        # 장애물 영역 내에 있다고 하면 충돌이라 판단
        for (cx, cy, r) in OBSTACLES:
            if hit_circleFunc(x, y, cx, cy, r + 0.25): # r은 본체 반경 포함시킴(넓게 가져가려고)
                return True
        # 영역 내에 있고 충돌 하지 않으면 False를 반환
        return False
    
    # 초기화 하는 함수
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 에이전트의 초기상태 설정하기
        self._state = np.array([2.0, 2.0, np.deg2rad(30.0)], dtype=np.float32)
        self._steps = 0
        self._prev_goal_dist = np.linalg.norm(self._state[:2] - self.goal) # 초기 목표 거리

        # obs와 info 호출
        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    
    # 에이전트가 행동 적용해 한 step씩 이동하는 함수 생성
    def step(self, action):
        self._steps += 1 # 현재 에피소드의 스텝
        x, y, th = self._state

        # 행동 적용하기
        if action == 0:             # 좌회전
            th += self.steer_delta  # +8도
        elif action == 2:           # 우회전
            th -= self.steer_delta  # -8도
        
        # th기준으로v만큼 x, y만큼 이동
        x += math.cos(th) * self.v 
        y += math.sin(th) * self.v

        # 이동 후 상태 갱신 - 새 위치, 새 방향으로 상태 갱신.
        self._state = np.array([x, y, th],dtype=np.float32)

        # 목표 거리 갱신 - 이동했으니까 목표거리가 바뀜
        goal_dist = np.linalg.norm(self._state[:2] - self.goal)
        # 접근 변화량 - 거리가 얼마나 변했는지
        process = self._prev_goal_dist - goal_dist
        # 다음 스텝을 위해서 현재 거리를 이전 거리로 저장하기
        self._prev_goal_dist = goal_dist

        # reward : 행동 결과에 대한 보상값
        # 보상값 계산하기 - 보상값의 변화를 주기위해 매스탭마다 패널티를 줘야함, 시간 낭비를 절약하기 위해 
        reward = 1.0 * process - 0.01 # 매 스탭마다 - 0.01 만큼 패널티 부여

        terminated, truncated = False, False
        # terminated : 종료 여부
        # truncated : 시간 제한
        # 목표 도달
        if goal_dist < self.goal_radius:
            reward += 1.0
            terminated = True
        # 충돌 여부
        if self._collision():
            reward -= 1.0
            terminated = True
        # step 초과
        if self._steps >= self.max_steps:
            terminated = True
        
        # 관측 정보 반환
        # obs : 새로운 상태 - 라이더 거리 벡터
        obs = self._get_obs()
        # info : 디버그 같은 추가 정보 - goal_dist, steps
        info= self._get_info()
        
        return obs, reward, terminated, truncated, info

    # 렌더링 하는 함수
    def render(self):
        if self.render_mode == 'human':
            print('현재 상태 :',self._state)
        
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(7.5, 5.5))
        
        ax = self.ax
        ax.clear
        ax.set_xlim(0, WORLD_W)
        ax.set_ylim(0, WORLD_H)
        ax.set_aspect('equal', adjustable='box') # 가로 세로 비율 동일하게(1:1), 왜곡 방지
        ax.set_title("Simple Lidar Env")
        ax.plot([0, WORLD_W, WORLD_W, 0, 0], [0, 0, WORLD_H, WORLD_H, 0], lw=2) # 경계 사각형
        
        # 장애물에 원그리기
        for (cx, cy, r) in OBSTACLES:
            circ = plt.Circle((cx, cy), r, edgecolor='tab:red', facecolor='none', lw=2)
            ax.add_patch(circ)
        goal = plt.Circle(tuple(self.goal), self.goal_radius, edgecolor='tab:blue', facecolor='none', lw=2)
        ax.add_patch(goal)

        # 에이전트 : 삼각형 형태로 그리기
        x, y, th = self._state
        L  = 0.6 # 삼각형의 길이(크기)

        tri = np.array([
            [x + np.cos(th) * L, y + np.sin(th) * L], # 삼각형의 제일 위 꼭짓점
            [x + np.cos(th + 2.5) * L / 1.5, y + np.sin(th + 2.5) * L / 1.5], # 왼쪽뒤 꼭짓점
            [x + np.cos(th - 2.5) * L / 1.5, y + np.sin(th - 2.5) * L / 1.5], # 오른쪽뒤 꼭짓점
        ])
        ax.fill(tri[:, 0], tri[:, 1], alpha=0.85, color='tab:blue', label='agent')

        # 라이다 빔 시각화
        obs, angs = cast_lidar(x, y, th)
        for d, a in zip(obs, angs):
            # 두 점을 연결하는 실질적인 선 그리기 실행
            ax.plot([x, x + np.cos(a) * d], [y, y + np.sin(a) * d], lw=1, alpha=0.9)
        ax.legend(loc='upper right')
        # show로 하면 한번만 보여주고 끝남. 
        # Frame 갱신 - rander() 내에서 매 번 그림이 다시 그려지기 위함. 
        plt.pause(0.05) 

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == "__main__":
    env = SimpleLiderEnv()
    obs, info = env.reset()

    # 강화학습에서 보상이 중요함.
    # (강화학습 보상값 : 에이전트가 목표에 달성하면 보상값을 주고 달성하지 못하면 패널티를 받음) 
    total_reward = 0.0

    # 최대 500 step반복
    for st in range(500): 
        # 환경에서 가능한 행동범위 중 무작위로 행동 하나를 선택함 (직, 좌, 우)
        # Gymnasium의 환경에서 자동으로 주어짐
        action = env.action_space.sample()
        # 환경 단계 실행(환경이 다음상태(=새로운 상태)와 보상 등의 반환값을 줌)
            # obs : 새로운 상태
            # reward : 행동 결과에 대한 보상값
            # terminated : 종료 여부
            # truncated : 시간 제한
            # info : 디버그 같은 추가 정보
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()

        # 에피소드가 끝나면 환경변수를 초기화 해야함 - 에피소드 종료조건을 줘야함
        if terminated or truncated:
            print(f'Episode and at step={st}, total_reward={total_reward:.3f}, info={info}')
            obs, info = env.reset()
            total_reward = 0.0

        # 모든 작업이 끝나면 자원을 반납해야한다
        env.close()