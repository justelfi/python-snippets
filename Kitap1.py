# -*- coding: utf-8 -*-
"""
Created on Wed May 14 09:33:33 2025

@author: Damla
"""

import pandas as pds
import matplotlib.pyplot as plt

# 1. Excel dosyasını oku, başlıkları 3. satırdan al (0-index ile skiprows=2)
df = pd.read_excel("C:\\Users\\Damla\\Desktop\\Kitap1.xlsx", skiprows=2)

# 2. Sütun adlarını anlamlı hale getir
df.columns = ["CoatingID", "Col2", "Col3", "Col4", "Col5", "Col6", "Time"]

# 3. Time sütununu datetime formatına çevir
df["Time"] = pd.to_datetime(df["Time"])

# 4. Belirli bir CoatingID’ye göre filtrele
kaplama1 = df[df["CoatingID"] == 2025032805]

# 5. Sonuçları yazdır
print(kaplama1.head())

plt.figure(figsize=(10, 6))

plt.plot(df['Time'], df['Col2'], label='Col2')
plt.plot(df['Time'], df['Col3'], label='Col3')
plt.plot(df['Time'], df['Col4'], label='Col4')
plt.plot(df['Time'], df['Col5'], label='Col5')
plt.plot(df['Time'], df['Col6'], label='Col6')

# Başlık, etiketler ve gösterim
plt.title('Kolon Verileri Zamanla Değişim Grafiği')
plt.xlabel('Zaman')
plt.ylabel('Değer')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Grafik göster
plt.show()