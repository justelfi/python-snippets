# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:03:06 2025

@author: Damla
"""

# Gerekli kütüphaneleri içe aktaralım
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %% Veri setini yükleyip, DataLoader oluşturalım

def get_data_loaders(batch_size=64):
    # Görselleri tensor'a çevirip normalize eden transform işlemi
    transform = transforms.Compose([
        transforms.ToTensor(),  # Görseli Tensor'a çevir
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB kanal başına normalize et
    ])

    # CIFAR-10 eğitim ve test veri setlerini indir ve dönüştür
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Veri setleri için DataLoader oluştur (batch_size kadar veriyi bir kerede çeker)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# %% Görselleri göstermek için yardımcı fonksiyonlar

def imshow(img):
    img = img / 2 + 0.5  # Normalizasyonu geri al
    np_img = img.numpy()  # Tensor'ı numpy array'e çevir
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # Görseli doğru şekilde çiz
    plt.show()

def get_sample_images(loader):
    data_iter = iter(loader)  # Verileri iterate edecek obje oluştur
    images, labels = next(data_iter)  # Bir batch veri al
    return images, labels

def visualize(n):
    # Eğitim verilerini yükleyelim
    train_loader, _ = get_data_loaders()

    # Eğitim verilerinden örnek batch al
    images, labels = get_sample_images(train_loader)

    # n adet resmi göster
    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i + 1)  # 1 satır n sütunluk gösterim
        imshow(images[i])  # Görseli göster
        plt.title(f"Label: {labels[i].item()}")  # Etiketini başlık olarak yaz
        plt.axis("off")  # Eksen çizgilerini kapat
    plt.show()

# visualize(5) diyerek örnek resimleri görebilirsin
visualize(5)

# %% CNN modelimizi oluşturalım

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # İlk katman: 3 kanal (RGB) → 32 filtre, kernel 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()  # Aktivasyon fonksiyonu
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 havuzlama
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 2. konvolüsyon katmanı
        self.dropout = nn.Dropout(0.2)  # %20 dropout
        self.fc1 = nn.Linear(64*8*8, 128)  # Tam bağlı katman
        self.fc2 = nn.Linear(128, 10)  # 10 sınıf olduğu için 10 çıkış

    def forward(self, x):
        # Veriyi sırasıyla katmanlardan geçir
        x = self.pool(self.relu(self.conv1(x)))  # 1. blok
        x = self.pool(self.relu(self.conv2(x)))  # 2. blok
        x = x.view(-1, 64*8*8)  # Flatten işlemi
        x = self.dropout(self.relu(self.fc1(x)))  # Tam bağlı + ReLU + Dropout
        x = self.fc2(x)  # Çıkış katmanı
        return x

# Cihazı seç (GPU varsa kullan yoksa CPU)
# en üste tasıdık device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli cihaza gönder
# model = CNN().to(device)

# %% Kayıp fonksiyonu ve optimizer tanımlayalım

def define_loss_and_optimizer(model):
    criterion = nn.CrossEntropyLoss()  # Çok sınıflı sınıflandırma için uygun
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Stokastik Gradient Descent
    return criterion, optimizer

# %% Modeli eğitme fonksiyonu

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()  # Eğitim moduna geç
    train_losses = []  # Kayıpları saklamak için liste

    for epoch in range(epochs):
        total_loss = 0  # Her epoch başında loss'u sıfırla
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Gradyanları sıfırla
            outputs = model(images)  # Modelden tahmin al
            loss = criterion(outputs, labels)  # Kayıp değeri hesapla
            loss.backward()  # Geri yayılım
            optimizer.step()  # Ağırlıkları güncelle

            total_loss += loss.item()  # Bu batch'in loss'unu ekle

        avg_loss = total_loss / len(train_loader)  # Ortalama epoch kaybı
        train_losses.append(avg_loss)  # Listeye ekle
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")  # Konsola yazdır

    # Eğitim kaybı grafiğini çiz
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker="o", label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

# %% Modeli çalıştıralım

train_loader, test_loader = get_data_loaders()  # Veriyi yükle
criterion, optimizer = define_loss_and_optimizer(model)  # Kayıp ve optimizer'ı tanımla
train_model(model, train_loader, criterion, optimizer, epochs=10)  # Eğitimi başlat


# %% test

def test_model(model, test_loader):
    
    model.eval()  # degerlendirme modu
    correct = 0   # dogru tahmin sayaci
    total = 0     # toplam veri sayaci
    
    with torch.no_grad():
        # gradyan hesaplamasini kapat
        for images, labels in test_loader:
            # test veri setini kullanarak degerlendirme
            images, labels = images.to(device), labels.to(device)  # verileri cihaza tasi
            
            outputs = model(images)  # prediction
            _, predicted = torch.max(outputs, 1)  # en yüksek olasilikli sinifi sec
            total += labels.size(0)  # toplam veri sayisi
            correct += (predicted == labels).sum().item()  # dogru tahminleri say
            
    print(f"{dataset_type} accuracy: {100 * correct / total} %")  # dogruluk oranini ekrana yazdir
    
#test_model(model, test_loader, dataset_type = "test")  # test accuracy:63.21 %
#test_model(model, train_loader, dataset_type= "training")  # training accuracy: 65.716 %

if __name__ == "__main__":
    
    # veri seti yukleme
    train_loader, test_loader = get_data_loaders()

    #gorsellestirme
    visualize(10)

    #training
    model = CNN().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs = 10)
    
    # test
    test_model(model, test_loader, dataset_type = "test")  # test accuracy: 63.21 %
    test_model(model, train_loader, dataset_type = "training") # training accuracy: 65.716 %
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    