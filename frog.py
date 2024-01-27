import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 초기화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 빈 그래프 생성
scatter = ax.scatter([], [], [], c='r', marker='o')

# 그래프 업데이트 함수 정의
def update_graph(frame):
    x, y, z, _ = frame
    scatter._offsets3d = (x, y, z)
    plt.draw()

# 동영상 프레임 값 리스트
frames = [
    [-2.22275322, 0.75831482, -0.42872753, 1.],
    [-1.17085557, 0.72417922, -0.50736713, 1.]
]

# 각 프레임을 순회하며 업데이트
for frame in frames:
    update_graph(frame)
    plt.pause(0.1)  # 잠시 대기하여 그래프가 업데이트되는 것을 확인

plt.show()
