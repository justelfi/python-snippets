
import numpy as np
import matplotlib.pyplot as plt

# Fonksiyon ve türevi
def func(x):
    return x**2

def grad(x):
    return 2*x

# Başlangıç noktası
x = 10  # rastgele bir yerden başlıyoruz
learning_rate = 0.1
epochs = 50

# Adımları kaydetmek için liste
x_history = [x]

# Gradient Descent döngüsü
for i in range(epochs):
    gradient = grad(x)
    x = x - learning_rate * gradient
    x_history.append(x)
    print(f"Adım {i+1}: x = {x:.4f}, f(x) = {func(x):.4f}")

# Grafikle gösterelim
x_vals = np.linspace(-12, 12, 100)
y_vals = func(x_vals)

plt.plot(x_vals, y_vals, label='y = x^2')
plt.scatter(x_history, [func(x) for x in x_history], color='red', label='Gradient Descent Adımları')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent ile Minimum Arayışı")
plt.legend()
plt.grid(True)
plt.show()

