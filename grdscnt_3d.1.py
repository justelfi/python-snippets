import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Kayıp fonksiyonu
def loss_fn(w0, w1):
    return np.sin(3 * w0) + np.cos(5 * w1)

# Gradyanlar (kısmi türevler)
def grad_loss(w0, w1):
    dw0 = 3 * np.cos(3 * w0)
    dw1 = -5 * np.sin(5 * w1)
    return dw0, dw1

# Başlangıç noktası
w0, w1 = 0.5, 0.5
learning_rate = 0.01
num_steps = 100
trajectory = [(w0, w1, loss_fn(w0, w1))]

# Gradient Descent Döngüsü
for _ in range(num_steps):
    dw0, dw1 = grad_loss(w0, w1)
    w0 -= learning_rate * dw0
    w1 -= learning_rate * dw1
    trajectory.append((w0, w1, loss_fn(w0, w1)))

# Meshgrid oluştur
w0_vals = np.linspace(0, 1, 100)
w1_vals = np.linspace(0, 1, 100)
W0, W1 = np.meshgrid(w0_vals, w1_vals)
Z = loss_fn(W0, W1)

# 3D çizim
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, Z, cmap='jet', alpha=0.7)

# Gradient descent adımları
traj = np.array(trajectory)
ax.plot(traj[:,0], traj[:,1], traj[:,2], color='black', marker='o', markersize=3, label='Descent Path')

ax.set_xlabel("w₀")
ax.set_ylabel("w₁")
ax.set_zlabel("J(w₀, w₁)")
ax.set_title("3D Gradient Descent Visualization")
ax.legend()
plt.show()
