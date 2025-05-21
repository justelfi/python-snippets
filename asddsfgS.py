import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ağırlıkları başlat
w = torch.randn(3, requires_grad=True)
eta = 0.1
steps = 50

trajectory = []

for step in range(steps):
    loss = (w[0] - 3)**2 + (w[1] + 2)**2 + (w[2] - 1)**2
    loss.backward()

    with torch.no_grad():
        trajectory.append((w[0].item(), w[1].item(), w[2].item(), loss.item()))
        w -= eta * w.grad

    w.grad.zero_()

# Görselleştirme
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x_vals = [pt[0] for pt in trajectory]
y_vals = [pt[1] for pt in trajectory]
z_vals = [pt[2] for pt in trajectory]

ax.plot(x_vals, y_vals, z_vals, marker='o', color='blue', label='Gradient Descent Path')
ax.scatter(x_vals[0], y_vals[0], z_vals[0], color='red', s=100, label='Start')
ax.scatter(3, -2, 1, color='green', s=100, label='Minimum (3, -2, 1)')

ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('w3')
ax.set_title('3D Gradient Descent Trajectory')
ax.legend()
plt.tight_layout()
plt.show()
