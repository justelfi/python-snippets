
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# 1. Veri Kümesi (Diyaloglar)
dialogs = [
    "bu dünya yalan mecnun",
    "leyla? o da kim?",
    "ben zaten hep buradaydım",
    "zamanı durdurdum leyla, gel dedim ama sen duymadın",
    "sana şiir yazdım leyla",
    "ben mecnun'um leyla için çöllere düşen",
    "beni unutma leyla, bu dünyada tek seninle olurdum",
    "gözlerim seni bekliyor leyla",
    "benim aşkımın adı dağlar, gökyüzü ve sen",
    "seninle her şey daha güzel leyla",
    "gönlümde tek sen varsın",
    "beni hep seveceksin leyla",
    "ayrılık çok acı, mecnun oldum seni her hatırladığımda",
    "sevgilim, her anı seninle yaşamak istiyorum"
]

# 2. Kelimeleri tokenize etme ve dizilere dönüştürme
tokenizer = {}
tokenizer_rev = {}
token_id = 1  # 0 indeksini rezerv edeceğiz
for dialog in dialogs:
    words = dialog.split()
    for word in words:
        if word not in tokenizer:
            tokenizer[word] = token_id
            tokenizer_rev[token_id] = word
            token_id += 1

# 3. Eğitim Verisini Hazırlama
def prepare_sequences(dialogs, tokenizer):
    sequences = []
    for dialog in dialogs:
        words = dialog.split()
        sequence = [tokenizer[word] for word in words]
        sequences.append(sequence)
    return sequences

sequences = prepare_sequences(dialogs, tokenizer)

# 4. LSTM Modeli
class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        return out

# Modeli Başlat
vocab_size = len(tokenizer) + 1  # 0'ı boşluk için ayırıyoruz
embedding_dim = 32
hidden_dim = 64
model = ChatbotModel(vocab_size, embedding_dim, hidden_dim)

# 5. Eğitim için Hazırlık
def prepare_data(sequences, window_size=2):
    x_data, y_data = [], []
    for seq in sequences:
        for i in range(len(seq)-window_size):
            x_data.append(seq[i:i+window_size])
            y_data.append(seq[i+window_size])
    return torch.tensor(x_data), torch.tensor(y_data)

window_size = 2
x_data, y_data = prepare_data(sequences, window_size)

# 6. Modeli Eğitmek
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Eğitim Döngüsü
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_data)
    loss = criterion(outputs.view(-1, vocab_size), y_data.view(-1))
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 8. Cümle Tamamlama Fonksiyonu
def generate_response(seed_text, model, tokenizer_rev, tokenizer, window_size=2):
    model.eval()
    words = seed_text.split()
    for _ in range(window_size):
        seq = [tokenizer.get(word, 0) for word in words[-window_size:]]
        seq = torch.tensor(seq).unsqueeze(0)
        output = model(seq)
        predicted_idx = torch.argmax(output, dim=2)[0, -1].item()
        predicted_word = tokenizer_rev.get(predicted_idx, '')
        words.append(predicted_word)
    return ' '.join(words)

# 9. Kullanım Örneği
if __name__ == "__main__":
    print(generate_response("ben zaten hep", model, tokenizer_rev, tokenizer))
    print(generate_response("leyla", model, tokenizer_rev, tokenizer))
