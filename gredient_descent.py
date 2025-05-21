import torch  #tensör işlemleri ve otomatik türev için
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt  #grafik çizimleri için
from mpl_toolkits.mplot3d import Axes3D #3d yüzey için

# Başlangıç noktası
w = torch.tensor([3.5, -3.5], requires_grad=True)  #w->iki boyutlu vetör, req=tensörün türevini hesaplar
learning_rate = 0.25 #adım büyüklüğü, her iterasyonda parametrenin ne kadar değişeceğini belirler
steps =80 #gd. algoritmasının kac kez calıstırılacagımız

# Kayıt listeleri
w0_list, w1_list, loss_list = [], [], [] #her adımda w0-w1 ve loss degerlerini bu listelere kaydeder ve bu sayede cizim yapabilir

# Gradient Descent
for _ in range(steps):
    loss = torch.sin(w[0]) * torch.cos(w[1]) + 0.1 * w[0]**2 + 0.1 * w[1]**2  #fonksiyonun minumumunu bulmak istiyoruz(sin-cos dalgali yüzey saglar+parabolik etki verir)
    loss.backward()  #loss fonksiyonunun w vektörüne göre türevini saglar

    with torch.no_grad():    #grd. hesaplamasını gecici olarak durdurur.Yoksa w değişimi kaydolur
        w -= learning_rate * w.grad  #gd. adımı burada yapılır. negatif gr. yönünde ilerlenir

    w.grad.zero_() #Eğer bu yapılmazsa, her backward() çağrısında gradient’ler birikmeye devam eder. Bu yüzden her adımda sıfırlıyor
    w0_list.append(w[0].item()) #.item() yöntemi ile tensör değerlerini Python sayısına çevirir
    w1_list.append(w[1].item())   #Listeye ekleyerek daha sonra grafikle göster
    loss_list.append(loss.item())

# Yüzey oluşturma
w0_range = torch.linspace(-4, 4, 100)  # -4 ile +4 arasında 100 değer üretir.
w1_range = torch.linspace(-4, 4, 100)  # 2D yüzeyin her noktasında (w0, w1) kombinasyonu oluşturur
W0, W1 = torch.meshgrid(w0_range, w1_range, indexing="ij")  #bu kombinasyonlara göre loss değerini hesaplar
Loss_surface = torch.sin(W0) * torch.cos(W1) + 0.1 * W0**2 + 0.1 * W1**2  

# 3D Plot
fig = plt.figure(figsize=(10, 7))   #3D figür oluştururuz ve eksenleri belirtiriz
ax = fig.add_subplot(111, projection="3d")  

ax.plot_surface(W0.numpy(), W1.numpy(), Loss_surface.numpy(), cmap="viridis", alpha=0.6)  #Yüzeyi 3D çizer
ax.plot(w0_list, w1_list, loss_list, color="red", marker="o", label="Gradient Descent")   #Gradient descent algoritmasının adım adım izlediği yolu gösterir

ax.set_xlabel("w₀")
ax.set_ylabel("w₁")
ax.set_zlabel("Loss")
ax.set_title("Karmaşık Fonksiyon Üzerinde Gradient Descent")
ax.legend()
plt.show()

#Bölüm	Ne Yapıyor?
#torch.tensor(...)	Başlangıç vektörü
#loss.backward()	Türev hesaplama
#w -= ...	Güncelleme (gradient descent)
#meshgrid	3D yüzey oluşturma
#plot_surface	Fonksiyon yüzeyi çizimi
#plot(...)	Optimizasyon yolunun çizimi
