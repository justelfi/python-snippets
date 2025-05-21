
# %% veri seti seçme

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def generate_data(seq_lenght=50, num_samples=1000):
    """ 
    example: 3lu paket
    sequence: [2,3,4] # giriş dizilerini saklamak için
    targets: [5] hedef değerleri saklamak için 
    """ 
    x = np.linspace(0, 100, num_samples)  # 0-100 arası num_samples kadar veri oluştur
    y = np.sin(x)
    sequence = []  # giriş dizilerini saklamak için
    targets = []   # hedef değerleri saklamak için

    for i in range(len(x) - seq_lenght):
        sequence.append(y[i:i+seq_lenght])  # input
        targets.append(y[i + seq_lenght])   # input dizisinden sonra gelen değer
  

        # Veriyi görselleştir
        x = np.linspace(0, 100, 1000)
        y = np.sin(x)
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='sin(t)', color='b', linewidth=2)
        plt.title('Sinüs Dalga Grafiği')
        plt.xlabel('Zaman (radyan)')
        plt.ylabel('Genlik')
        plt.legend()
        plt.grid(True)
        plt.show()

        return np.array(sequence), np.array(targets)

sequence, targets = generate_data()    


# %% rnn modelini olustur

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        """ 
            RNN -> Linear (output)
        """
        
        super (RNN, self).__init__()
        # input_size: giris boyutu
        # hidden_size: rnn gizli katman cell sayisi
        # num_layers: rnn layer sayisi
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)  # RNN layer
        # output_size: cikti boyutu
        self.fc = nn.Linear(hidden_size, output_size)  # fully connected layer: output 


    def forward(self, x):
        
        out,_ = self.rnn(x)  # rnn e girdiyi ver ciktiyi al
        out = self.fc(out[:,-1, :])  # son zaman adimindaki ciktiyi al ve fc layera bagla
        return out
    
model = RNN(1, 16, 1, 1)

# %% rnn training
    
# hyperparameters
seq_length = 50  # input dizisinin boyutu
input_size = 1  # input dizisinin boyutu
hidden_size = 16  # rnn in gizli katmandaki dugum sayisi
output_size = 1  # output boyutu yada tahmin edilen deger
num_layers = 1  # rnn katman sayisi
epochs = 20  # modelin kaç kez tum veri seti üzerinde egitilecegi
batch_size = 32  # her bir egitim adiminda kac örnegin kullanılacagi
learning_rate= 0.001  # optimizasyon algoritmai icin ogrenme orani icin ogrenme orani yada hizi

# veriyi hazirla
X, y = generate_data(seq_length)
X = torch.tensor(X, dtype = torch.float32).unsqueeze(-1)  # pytorch tensorune cevir ve boyut eklee
y = torch.tensor(y, dtype = torch.float32).unsqueeze(-1)  # pytorch tensorune cevir ve boyut ekle

dataset = torch.utils.data.TensorDataset(X, y)  # pytorch dtaset olusturma
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  #veri yukleyici olustur

#modeli tanimla
model = RNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()  # loss function: mean square error- ortalama kare hesabı
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimization = adaptive momentum

for epoch in range(epochs):
    for batch_x, batch_y in dataLoader:
        optimizer.zero_grad()  # gradyanlari sifirla
        pred_y = model(batch_x)  # modelden tahmini al
        loss = criterion(pred_y, batch_y)  # model tahmini olani karsilastir ve loss hesapla
        loss.backward()  # geri yayilim ile gradyanlari hesapla
        optimizer.step()  # agirliklari hesapla
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
# %% rnn test and evulation

#test için veri olustur
X_test = np.linspace(100, 110, seq_length).reshape(1,-1)  # ilk test verisi
y_test = np.sin(X_test)  # test verilerimizin gercek degeri

X_test2 = np.linspace(120, 130, seq_length).reshape(1,-1)  # ikinci test verisi
y_test2 = np.sin(X_test2)   

# from numpy to tensor
X_test = torch.tensor(y_test, dtype = torch.float32).unsqueeze(-1)
X_test2 = torch.tensor(y_test2, dtype = torch.float32).unsqueeze(-1)

# modeli kullanarak prediction yap
model.eval()
prediction1 = model(X_test).detach().numpy()  # ilk test verisi  icin tahmin
prediction2 = model(X_test2).detach().numpy()

#sonucları gorsellestir

plt.figure()
plt.plot(np.linspace(0, 100, len(y)), y, marker = "o", label = "Training dataset")
plt.plot(X_test.numpy().flatten(), marker = "o", label = "Test 1") 
plt.plot(X_test2.numpy().flatten(), marker = "o", label = "Test 2")
plt.plot(np.arange(seq_length, seq_length + 1), prediction1.flatten(), "ro", label = "Prediction 1")
plt.plot(np.arange(seq_length, seq_length + 1), prediction2.flatten(), "ro", label = "Prediction 2")
plt.legend()
plt.show()
   
    
    
    
    
    
    
    
    

