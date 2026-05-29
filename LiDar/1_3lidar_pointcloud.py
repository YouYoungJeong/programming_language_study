'''
자동차가 이동하며 여러 방향으로 레이저를 쏘고 물체(건물) 표면 좌표를 수집하여 3D점군(Point Cloud) 생성
라이다는 물체를 면으로 보는 것이 아니라 많은 점좌표(x, y)를 모아 환경을 재구성 한다.
즉, [x, y, z]이러 점으로 구성 할 수 있다.
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3d 이미지 출력

# 도시 환경 정의 : 각 건물은 직육면체 형태 (xmin, xmax, ymin, ymax, zmin, zmax)
building = [
    (-20, -10, 10, 20, 0, 20),
    (10, 20, 15, 25, 0, 25),
    (-15, -5, 35, 45, 0, 18),
    (5, 18, 50, 60, 0, 30),
]

# 자동차 이동 경로
car_positions = []

# 자동차가 y축 방향으로 이동 한다고 가정. y=0 ~ y=60 구간을 25개 위치로 나눔
for y in np.linspace(0, 60, 25):
    # 도로에 건물이 있는데 차량이 건물 사이를 지나게 함.
    # -> [x:차랑위치, y:현재 전진 위치, z:센서 높이]
    car_positions.append(np.array([0, y, 2])) 
    # print(f'car positions : {car_positions}')
    
# LiDAR스캔 함수
def simulat_lidar(car_pos):
    points = []
    horizental_angles = np.linspace(-90, 90, 120) # 수평(좌우)방향 스캔 각도 (-90도 ~ 90도 120개)
    vertical_angles = np.linspace(-15, 15, 8) # 수직(상하)방향 스캔 각도 (-15도 ~ 15도 8개)
    max_distance = 80
    # 모든 방향으로 레이저 발사
    for h_deg in horizental_angles:
        for v_deg in vertical_angles:
            h = np.radians(h_deg) # degree를 radian으로 변환
            v = np.radians(v_deg) # degree를 radian으로 변환

            # 레이저 방향 벡터 계산
            dx = np.cos(v) * np.sin(h)
            dy = np.cos(v) * np.cos(h)
            dz = np.sin(v)

            # 레이저 방향으로 진행
            for d in np.linspace(0, max_distance, 400):
                # 현재 레이저의 위치를 계산
                x = car_pos[0] + dx * d
                y = car_pos[1] + dy * d
                z = car_pos[2] + dz * d

                hit = False
                
                # 모든 건물에 충돌 검사 하기
                for b in building:
                    xmin, xmax, ymin, ymax, zmin, zmax = b
                    # 현재 레이저 위치가 건물 내부에 있는지 판단
                    inside = (xmin <= x <= xmax and ymin <=y <= ymax and zmin <= z <= zmax)

                    # 레이저가 건물과 충돌한 경우
                    if inside:
                        points.append([x, y, z])
                        hit = True
                    
                if hit: break
    return points

# point cloud 저장용 리스트
all_points = []

for pos in car_positions:
    scan_points = simulat_lidar(pos)
    all_points.extend(scan_points)

all_points = np.array(all_points) # numpy 배열로 변환
print(all_points) # [[-19.59775284  10.08350691   1.17533206] ...

# Point Cloud 시각화
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    all_points[:, 0],   # x 좌표
    all_points[:, 1],   # y 좌표
    all_points[:, 2],   # z 좌표
    s = 1,
    c = all_points[:, 2],
    cmap = 'jet'
)

car_positions_np = np.array(car_positions)
# 자동차 이동 경로 표시
ax.plot(
    car_positions_np[:, 0], # x 좌표
    car_positions_np[:, 1], # y 좌표
    car_positions_np[:, 2], # z 좌표
    c = 'black',
    linewidth = 3,
    label = 'Car Path',
)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Simple LiDAR Point Cloud')
ax.set_xlim(-40, 40)
ax.set_ylim(0, 80)
ax.set_zlim(0, 20)
plt.legend()
plt.show()