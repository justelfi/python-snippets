"""
Problem tanimi: mnist veri seti ile rakam sınıflandırma projesi
MNIST
ANN: Yapay Sinir Aglari

"""

#%%library
import torch #pytorch kutuphanesi, tensor islemleri
import torch.nn as nn #yapay sinir agi atmanlarini tanimlamak icin kullan
import torch.optim as optim #optimizasyon algoritmalarini iceren modul
import torchvision #görüntü isleme ve pre-defined modelleri icerir
import torchvision.transforms as transforms #görüntü dönüsümleri yapmak
import matplotlib.pyplot as plt #görsellestirme
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.plot([1, 2, 3], [4, 5, 1])
plt.title("Basit Grafik")
plt.show()


#optional: cihazı belirle gpu vs cpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#veri seti yükleme, data loading
def get_data_loaders(batch_size=32): #her iterasyonda islenecek veri miktari, batch size
    transform = transforms.Compose([
        transforms.ToTensor(),                     #görüntüyü tensore cevirir
        transforms.Normalize((0.5,),(0.5,)) #pixel degerlerini -1 ile 1 arasında ölcekler
        ])
    #mnist veri setini indir ve egitim test kümelerini olustur
    train_set = torchvision.datasets.MNIST(root="./data", train = True, download=True, transform = transform)
    test_set = torchvision.datasets.MNIST(root = "./data", train = False, download=True, transform = transform)
 
    #pytorch veri yukleyicisini olustur
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle = False)
    
    return train_loader, test_loader
train_loader, test_loader = get_data_loaders()
 
#data visualization
def visualize_samples(loader, n=5):
    images, labels = next(iter(loader)) #ilk batch den goruntu etiketleri alalim
    print(images[0].shape)
    fig, axws = plt.subplots(1, n, figsize=(10,5)) #2 n farklı goruntu icin gorsellestirme alanı
    for i in range(n):
        axws[i].imshow(images[i].squeeze(), cmap = "gray")  #gorseli gri tonlamali olarak goster
        axws[i].set_title(f"Label: {labels[i].item()}") #goruntute ait sinif etiketini baslik olarak yaz
        axws[i].axis("off") #eksenleri gizle
    plt.show()
    
visualize_samples(train_loader, )

#%%define ann model

#yapay sinir agi class
class NeuralNetwork(nn.Module): #pytorch un nn.module sinifindan miras alıyor
    def __init__(self):
        super(NeuralNetwork,self).__init__()
       
        self.flatten = nn.Flatten() #elimizde bulunan goruntuleri (2d) vektör halinde cevirelim(1d)
        
        self.fc1 = nn.Linear(28*28, 128) #ilk tam bagli katmani olustur: 784=input size, 128=outpu size
        
        self.relu = nn.ReLU() # aktivasyon fonksiyonu olustur
        
        self.fc2 = nn.Linear(128, 64) # ikinci tam bagli katmani olustur:128=input size, 64=output size
        
        self.fc3 = nn.Linear(64, 10) # citi katmani olustur: 64 =input size, 10=output size (0-9 etiketleri)
        
    def forward(self, x): 
       # forward propagation: ileri yayilim, giris olarak x=gorüntü alsin
   
       x=self.flatten(x) # inital x =28*28 lik bir görüntü -> duzlestir 784 vektör haline getir
       x=self.fc1(x) #birinci bagli katman
       x=self.relu(x) #aktivasyon fonksiyonu
       x=self.fc2(x) #ikinci bagli katman
       x=self.relu(x) #aktivasyon fonksiyonu
       x=self.fc3(x) #output katmanı
       
       return x # modelimizin ciktisini return edelim
   
      
        
        #ilk tam bagli katmani olustur
        
        #aktivasyon fonksiyonu olustur
        
        #cikti katmani olustur
        
        


#create model and compile

model = NeuralNetwork().to(device)

#kayip fonksiyonu ve optimizasyon algoritmasini belirle
define_loss_and_optimizer = lambda model:(
    nn.CrossEntropyLoss(),         #multi class classification problems loss function
    optim.Adam(model.parameters(), lr = 0.001)     #update weights with adam
    )

criterion, optimizer = define_loss_and_optimizer(model)


#%%train
def train_model(model, train_loader, criterion, optimizer, epochs = 10):
    
    model.train ()#modelimizi egitim moduna alalim
    train_losses = [] #her bir epoch sonucunda elde edilen loss degerlerini saklamak icin bir liste
    
    for epoch in range(epochs): #belirtilen epoch sayisi kadar egitim yapalim
        total_loss = 0  #toplam kayip degeri
   
        for images, labels in train_loader: #tüm egitim verileri uzerinde iterasyon gerceklesir
            images, labels = images.to(device), labels.to(device)  #verileri cihaza tasi
   
            optimizer.zero_grad() #gradyanlari sifirla
              
            predictions = model(images) #modeli uygula, forward pro.
            loss = criterion(predictions, labels) #loss hesaplama -> y_prediction ile y_real
            loss.backward()   #geri yayilim yani gradyan hesaplama
            optimizer.step()   #update weights (agirliklari guncelle)
        
        avg_loss = total_loss / len(train_loader) #ortalama kayip hesaplama
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")
  
    #loss graph
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
    
train_model(model, train_loader, criterion, optimizer, epochs=1)

#%%test

def test_model(model,test_loader):
    model.eval() #modelimizi değerlendirme moduna al
    correct = 0 #doğru tahmin sayaci
    total = 0 #toplam veri sayaci
    
    
    with torch.no_grad(): #gradyan hesaplama  ereksiz olduğundan kapattık
        for images, labels in test_loader: #test veri kümesini donguye al
            images, labels = images.to(device), labels.to(device)  #verileri cihaza tasi
            predictions = model(images)
            _, predicted = torch.max(predictions, 1) #en yüksek olasilikli sinifin etiketini bul
            total += labels.size(0) #toplam veri sayisini guncelle
            correct += (predicted == labels).sum().item() #dogru tahminleri say
            
    print(f"Test Accuracy : {100*correct/total:.3f}%")

test_model(model, test_loader)

# %%main

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders() #veri yukleyicilerini al
    visualize_samples(train_loader, 5)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer)
    test_model(model, test_loader)