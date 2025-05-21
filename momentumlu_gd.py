import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Başlangıç ayarları
w = torch.tensor([2.5, -2.5], requires_grad=True)
learning_rate = 0.1
momentum_factor = 0.9
velocity = torch.zeros_like(w)  # Başlangıçta hız sıfır

steps = 80
w0_list, w1_list, loss_list = [], [], []

# Gradient Descent + Momentum
for _ in range(steps):
    loss = torch.sin(w[0]) * torch.cos(w[1]) + 0.1 * w[0]**2 + 0.1 * w[1]**2
    loss.backward()

    with torch.no_grad():
        velocity = momentum_factor * velocity - learning_rate * w.grad  # momentumlu hız
        w += velocity  # hızla güncellenen w

    w.grad.zero_()
    w0_list.append(w[0].item())
    w1_list.append(w[1].item())
    loss_list.append(loss.item())

# Yüzey oluşturma
w0_range = torch.linspace(-4, 4, 100)
w1_range = torch.linspace(-4, 4, 100)
W0, W1 = torch.meshgrid(w0_range, w1_range, indexing="ij")
Loss_surface = torch.sin(W0) * torch.cos(W1) + 0.1 * W0**2 + 0.1 * W1**2

# 3D grafik
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0.numpy(), W1.numpy(), Loss_surface.numpy(), cmap='viridis', alpha=0.6)

line, = ax.plot([], [], [], color='orange', marker='o', label="Momentum Descent")
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(torch.min(Loss_surface).item(), torch.max(Loss_surface).item())
ax.set_xlabel("w₀")
ax.set_ylabel("w₁")
ax.set_zlabel("Loss")
ax.set_title("Momentumlu Gradient Descent Animasyonu")
ax.legend()

# Animasyon fonksiyonu
def update(frame):
    line.set_data(w0_list[:frame], w1_list[:frame])
    line.set_3d_properties(loss_list[:frame])
    return line,

ani = FuncAnimation(fig, update, frames=len(w0_list), interval=100, blit=False)

plt.show()

