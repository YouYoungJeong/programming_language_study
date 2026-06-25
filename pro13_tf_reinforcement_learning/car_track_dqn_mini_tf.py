"""
MiniCar DQN - Pretrain (TF/Keras)
- Straight lane, 5 LiDAR + speed (obs=6), 4 actions, Double DQN, pygame HUD.
- Aggressive early episodes, Finish line (thickness=1).
"""
import math, random, sys, time
import numpy as np, pygame
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Config
class Cfg:
    SEED=42 
    WIDTH,HEIGHT=960,640
    LANE_W,MARGIN=120,60
    SC,FOV,SMAX=5,120,220.0
    V_MAX,MAX_STEER=1.6,math.radians(24)
    MAX_ACCEL,FRIC,TURN=0.06,0.014,0.062
    OBS,N_ACT,HID=SC+1,4,128
    GAMMA,LR,BATCH,BUF=0.99,2.5e-4,128,50_000
    START,SYNC,WARM=800,500,30

    # EPISODES = 240 → 전체 학습 에피소드 수. 즉, train() 함수의 이 루프에서 240번 반복.
    # 한 에피소드 안에서 최대 500 step까지 실행된다.
    # 즉, 자동차가 부딪히거나(CRASH/OUT), 도착선 통과(FINISH), 타임아웃(TIMEOUT)되기 전까지 최대 500 스텝 동안 달릴 수 있다.
    # 총 240 에피소드 × 에피소드당 최대 500스텝
    # → 약 최대 12만 스텝(= 240×500) 동안 강화학습을 수행한다.(단, 충돌·완주 시 조기 종료됨)
    EP_STEPS, EPISODES=500, 240 
    EPS0,EPS1,EDECAY=0.9,0.05,20_000
    EARLY_AGGR_EP=40
    FPS,HEADLESS=60,False

random.seed(Cfg.SEED); np.random.seed(Cfg.SEED); tf.random.set_seed(Cfg.SEED)

# Env (finish line 포함)
class Env:
    def __init__(self):
        cx = Cfg.WIDTH/2
        self.l,self.r=cx-Cfg.LANE_W/2, cx+Cfg.LANE_W/2
        self.t,self.b=Cfg.MARGIN,Cfg.HEIGHT-Cfg.MARGIN
        self.finish_y=self.b-24
        self.reset()

    def reset(self):
        near_wall = (random.random() < 0.7)
        if near_wall:
            side = random.choice([-1, 1])
            edge_x = self.l + 12 if side<0 else self.r - 12      
            self.x = edge_x + np.random.uniform(-4,4)    # 난수폭 ±4
        else:
            self.x = (self.l + self.r)/2 + np.random.uniform(-10,10)
        self.y=self.t + 22 + np.random.uniform(-4,4)
        self.a=math.pi/2 + np.random.uniform(-math.radians(8), math.radians(8))
        self.v=0.0; self.k=0
        self.fail=""
        self.prev=np.array([self.x, self.y], np.float32)
        return self.obs()

    def _on(self, x, y):
        eps = 1.5    
        return (self.t-eps <= y <= self.b+eps) and (self.l-eps <= x <= self.r+eps)

    # 센서 감지 함수
    # 즉, 라이다는 차량의 관측(Observation) 입력으로 직접 쓰이고,
    # 이 값들을 바탕으로 DQN이 왼쪽/직진/오른쪽/브레이크 행동을 학습하게 된다.
    # 시작점 (px, py)에서 방향 ang으로 레이를 쏴 트랙 벽(왼쪽·오른쪽·위·아래)에 부딪히는 거리(best)를 구함.
    # 즉, 실제 라이다 센서가 '이 방향으로 얼마만큼 앞이 비어 있는가?'를 계산하는 역할.
    # 반환값은 0~Cfg.SMAX 범위 거리값이다. (기본적으로 최대 탐지 거리 220픽셀)
    def _ray(self,px,py,ang,dmax=Cfg.SMAX):
        dx, dy, best=math.cos(ang), math.sin(ang), dmax
        if abs(dx) > 1e-9:
            for X in (self.l,self.r):
                t=(X-px)/dx
                if 0<=t<=dmax:
                    yy=py + t * dy
                    if self.t <= yy <= self.b: best = min(best, t)
        if abs(dy)>1e-9:
            for Y in (self.t, self.b):
                t=(Y-py) / dy
                if 0 <=t <= dmax:
                    xx = px + t * dx
                    if self.l <= xx <= self.r: best=min(best, t)
        return best

    # 센서 감지 함수
    # Cfg.SC = 5이므로, 총 5개의 라이다 레이를 120° 시야(Cfg.FOV)로 쏜다.
    # 각 레이 방향은 자동차의 진행 각도(self.a) 기준으로 좌우로 분포.
    # _ray()로부터 얻은 거리값을 Cfg.SMAX으로 나눠 0~1 정규화한 결과를 반환.
    def sensors(self):
        span=math.radians(Cfg.FOV)
        return np.array([self._ray(self.x, self.y, self.a + (-span/2 + span * i / (Cfg.SC-1))) / Cfg.SMAX for i in range(Cfg.SC)], np.float32)

    def obs(self):  # 관측값(observation) 생성
        # 센서값(5개) + 속도 1개 → 총 6차원 벡터(observation)
        # 즉, DQN의 입력(Cfg.OBS = 6)은 [라이다 5개 거리 + 속도 1개]
        # 예 : [0.83, 0.66, 0.42, 0.65, 0.85, 0.21] → 차 주변 장애물 거리 5개 + 현재 속도(정규화)
        # draw()에서 시각화
        return np.concatenate([self.sensors(),[self.v / Cfg.V_MAX]]).astype(np.float32)

    def step(self,act, thr_scale=1.0, aggressive=False):
        steer=[-1,0,1,0][act]
        base_thr=[1,1,1,-1][act]
        s=self.sensors()
        left,mid,right=s[0],s[2],s[-1]
        # throttle policy
        if aggressive:
            thr_eff = base_thr * thr_scale
            if mid<0.02: thr_eff = -1.0
        else:
            thr_eff = base_thr * thr_scale * (0.5 + 0.5 * mid)
            if mid < 0.08: thr_eff = -1.0
        # kinematics
        prev_y=self.y
        self.v=float(np.clip(self.v+thr_eff * Cfg.MAX_ACCEL,0.0, Cfg.V_MAX))
        self.a+=np.clip(steer, -1,1) * Cfg.MAX_STEER * (1 + 0.15 * self.v) * Cfg.TURN
        self.v=max(0.0, self.v-Cfg.FRIC) if self.v > 1e-6 else 0.0
        self.x+=math.cos(self.a) * self.v
        self.y+=math.sin(self.a) * self.v
        if not self._on(self.x, self.y): self.fail="CRASH/OUT"

        # reward
        mn=float(np.min(s))
        cur=np.array([self.x,self.y], np.float32)
        prog=float(np.dot(cur-self.prev, np.array([math.cos(self.a), math.sin(self.a)], np.float32))) 
        self.prev=cur
        r=0.12 + 0.07 * max(0.0, prog) - 0.02 * max(0.0, -prog) + 0.52 * mid + 0.16 * (1 - abs(right - left)) - 0.9 * max(0.0, 0.25 - min(left, mid, right))
        
        if act == 3: 
            r += (0.10 if mid < 0.12 else -0.05)

        # done: finish / crash / timeout
        done=False
        if (prev_y < self.finish_y <= self.y) and (self.l <= self.x <= self.r) and not self.fail:
            self.fail="FINISH"
            done=True

        if mn < 0.02 or self.fail=="CRASH/OUT": 
            r -= 25.0
            done=True

        self.k+=1
        if self.k >= Cfg.EP_STEPS and not done:
            self.fail=self.fail or "TIMEOUT"
            done=True

        return self.obs(), r, done

# Render (finish line + HUD)
def init_screen():
    pygame.init()
    scr=pygame.display.set_mode((Cfg.WIDTH, Cfg.HEIGHT))
    pygame.display.set_caption("MiniCar Pretrain - TF + finish")
    f=pygame.font.SysFont("consolas", 18)
    return scr, pygame.time.Clock(), f

# 게임 화면에 라이다 5개를 노란색 선으로 표시해 준다.
# 실시간으로 차량이 주변 벽에 가까워질수록 선이 짧게 보인다.
def draw(scr, env, font, hud):
    scr.fill((18,18,22))
    # lane
    pygame.draw.rect(scr, (70,70,70), (env.l, env.t, Cfg.LANE_W, env.b-env.t))
    pygame.draw.line(scr, (220,220,220), (env.l, env.t), (env.l,env.b),2)
    pygame.draw.line(scr, (220,220,220), (env.r, env.t), (env.r,env.b),2)
    # finish line (thickness=1)
    pygame.draw.line(scr, 
        (240,240,240), (env.l, env.finish_y), (env.r, env.finish_y),1)
    # car
    L, W = 26, 14
    ca, sa=math.cos(env.a), math.sin(env.a)
    p1=(env.x + (L / 2) * ca,env.y + (L / 2) * sa)
    p2=(env.x + (-L / 2) * ca + (-W / 2) * (-sa), env.y + (-L / 2) * sa + (-W / 2) * (ca))
    p3=(env.x + (-L / 2) * ca + ( W / 2) * (-sa), env.y + (-L / 2) * sa + ( W / 2) * (ca))
    pygame.draw.polygon(scr, (0,170,255), [p1,p2,p3])
    # sensors
    span=math.radians(Cfg.FOV)
    for i in range(Cfg.SC):
        ang=env.a+(-span / 2 + span * i / (Cfg.SC - 1))
        d=env._ray(env.x, env.y, ang, Cfg.SMAX)
        pygame.draw.line(scr,(255,200,50), (env.x, env.y), \
            (env.x + math.cos(ang) * d, env.y + math.sin(ang) * d), 1)
    # HUD
    px, pw=Cfg.WIDTH - 260, 250
    pygame.draw.rect(scr, (28,28,34), (px,12,pw,150))
    lines=[
        f"Ep: {hud['ep']} / {Cfg.EPISODES}",
        f"Step: {hud['gstep']}",
        f"Return: {hud['ret']:.1f}",
        f"Avg10: {hud['avg10']:.1f}",
        f"Eps: {hud['eps']:.2f}",
        f"Alive: {hud['alive']}",
        f"Fail10: crash {hud['cr10']} / to {hud['to10']}",
        f"Status: {hud['status']}",
    ]
    y=20
    for t in lines: 
        scr.blit(font.render(t, True, (230,230,230)), (px + 10, y))
        y += 20


# DQN (TF/Keras)
class QNet(keras.Model):
    def __init__(self, in_dim, out_dim, hidden=Cfg.HID):
        super().__init__()
        self.d1 = layers.Dense(hidden, activation="relu")
        self.d2 = layers.Dense(hidden, activation="relu")
        self.out = layers.Dense(out_dim, activation=None)

    def call(self, x, training=False):
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)

class Replay:
    def __init__(self, cap):
        self.cap=cap
        self.ptr=0
        self.full=False
        self.S=np.zeros((cap, Cfg.OBS), np.float32)
        self.A=np.zeros((cap,), np.int32)
        self.R=np.zeros((cap,), np.float32)
        self.N=np.zeros((cap, Cfg.OBS), np.float32)
        self.D=np.zeros((cap,), np.float32)

    def __len__(self): 
        return self.cap if self.full else self.ptr

    def push(self,s,a,r,n,d):
        i=self.ptr
        self.S[i]=s 
        self.A[i]=a 
        self.R[i]=r 
        self.N[i]=n 
        self.D[i]=float(d)
        self.ptr=(i + 1) % self.cap
        self.full = self.full or self.ptr==0

    def sample(self,b):
        idx=np.random.randint(0, len(self), size=b)
        return (self.S[idx], self.A[idx], self.R[idx], \
                self.N[idx], self.D[idx])

def eps(step):
    t=min(1.0, step / Cfg.EDECAY)
    return Cfg.EPS0 + (Cfg.EPS1 - Cfg.EPS0) * t

@tf.function
def train_step(online, target, opt, bs, ba, br, bn, bd):
    with tf.GradientTape() as tape:
        q_all = online(bs, training=True)                         # (B,A)
        idx = tf.stack([tf.range(tf.shape(ba)[0], dtype=tf.int32), ba], \
                       axis=1)
        q_sa = tf.gather_nd(q_all, idx)
        next_q_online = online(bn, training=False)
        next_act = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
        next_q_target = target(bn, training=False)
        next_idx = tf.stack([tf.range(tf.shape(next_act)[0], dtype=tf.int32), next_act], axis=1)
        next_q = tf.gather_nd(next_q_target, next_idx)
        target_q = br + (1.0 - bd) * Cfg.GAMMA * next_q
        loss = tf.keras.losses.Huber()(target_q, q_sa)

    grads = tape.gradient(loss, online.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    opt.apply_gradients(zip(grads, online.trainable_variables))
    return loss

# Train
def train():
    env=Env()
    q = QNet(Cfg.OBS, Cfg.N_ACT) 
    q(tf.zeros((1, Cfg.OBS)))
    tq = QNet(Cfg.OBS, Cfg.N_ACT)
    tq(tf.zeros((1, Cfg.OBS)))
    tq.set_weights(q.get_weights())
    opt = keras.optimizers.Adam(learning_rate = Cfg.LR)
    buf = Replay(Cfg.BUF)

    if not Cfg.HEADLESS: 
        scr, clk, font=init_screen()

    g=0; returns=[]; fails=[]

    for ep in range(1, Cfg.EPISODES + 1):
        s=env.reset()
        ret=0.0
        done=False
        aggressive = (ep <= Cfg.EARLY_AGGR_EP)
        thr_scale = 1.0 if aggressive else \
            (0.7 + min(1.0, (ep-Cfg.EARLY_AGGR_EP) / 80.0) * 0.3)

        # warm-up
        for _ in range(Cfg.WARM):
            a=np.random.randint(0, Cfg.N_ACT)
            ns,r,d=env.step(a,thr_scale, aggressive)
            buf.push(s, a, r, ns, d)
            s=ns
            ret+=r
            if d: s, ret=env.reset(), 0.0

        while not done:
            if not Cfg.HEADLESS:
                for e in pygame.event.get():
                    if e.type==pygame.QUIT: 
                            pygame.quit()
                            sys.exit(0)

            if random.random() < eps(g):
                a = np.random.randint(0, Cfg.N_ACT)
            else:
                qs = q(tf.convert_to_tensor(s[None, :], dtype=tf.float32), training=False)
                a = int(tf.argmax(qs, axis=1).numpy()[0])

            ns, r, done=env.step(a, thr_scale, aggressive)
            buf.push(s, a, r,ns, done)
            s = ns;  ret += r;   g += 1

            if len(buf) >= Cfg.START:
                bs, ba, br, bn, bd = buf.sample(Cfg.BATCH)
                bs=tf.convert_to_tensor(bs, tf.float32)
                ba=tf.convert_to_tensor(ba, tf.int32)
                br=tf.convert_to_tensor(br, tf.float32)
                bn=tf.convert_to_tensor(bn, tf.float32)
                bd=tf.convert_to_tensor(bd, tf.float32)
                _ = train_step(q, tq, opt, bs, ba, br, bn, bd)
                if g % Cfg.SYNC == 0: 
                      tq.set_weights(q.get_weights())

            if not Cfg.HEADLESS:
                cr10=sum(1 for x in fails[-10:] if x=="CRASH/OUT")
                to10=sum(1 for x in fails[-10:] if x=="TIMEOUT")
                draw(scr, env, font, {"ep":ep,"gstep":g, "ret":ret, 
                        "avg10": float(np.mean(returns[-10:])) if returns else 0.0,
                        "eps":eps(g), "alive":env.k, "cr10":cr10, "to10": to10,
                        "status":env.fail or ("AGGR" if aggressive else "RUN")
                  }
                )
                pygame.display.flip()
                clk.tick(Cfg.FPS)

        returns.append(ret); fails.append(env.fail or "TIMEOUT")
        if ep%10==0:
            print(f"[Ep {ep:04d}] avg10={np.mean(returns[-10:]):.1f}, \
                    crash10={sum(1 for x in fails[-10:] if x=='CRASH/OUT')}")

    # 학습이 끝난 후 DQN 모델의 가중치(weight)를 파일로 저장
    # q → 학습 중이던 온라인 Q-네트워크(QNet)
    q.save_weights("mini_pretrain_tf.weights.keras")
    print("Saved: mini_pretrain_tf.weights.keras")

    if not Cfg.HEADLESS:
        t0=time.time()
        while time.time() - t0 < 0.6:
            for e in pygame.event.get():
                if e.type==pygame.QUIT: 
                     pygame.quit()
                     sys.exit(0)
            time.sleep(0.02)
            pygame.quit()

if __name__=="__main__":
    train()

# 지금 설정으로는 학습이 진행될수록 피니시 라인 도달 빈도가 증가하고, 좌우 흔들림은 줄지만 아주 약간의 S-형 보정은 남는 게 정상임.

"""
추후 추가 학습이나 주행 테스트를 하려면 이렇게 불러올 수 있다.
q = QNet(Cfg.OBS, Cfg.N_ACT)
q(tf.zeros((1, Cfg.OBS)))      # 입력 한번 호출로 네트워크 초기화
q.load_weights("mini_pretrain_tf.weights.keras")
print("Model weights loaded successfully.")

load_weights()는 저장된 가중치만 복원하므로 네트워크 구조(QNet)는 동일하게 선언해둬야 함.
"""