liste = [1, 4, 6, 7, 5, 8]  # Listeyi doğru şekilde tanımladık
liste2 = [50, 40, 10, 20]

def ortalamahesapla(x):
    sayac = 0
    uzunluk = len(x)  # Uzunluğu sadece bir kez hesapla
    for i in x:
        sayac += i  # Sayacı artır
    ortalama = sayac / uzunluk  # Ortalama hesapla
    print(ortalama)

# İlk listeyi fonksiyona gönder
ortalamahesapla(liste)

# İkinci listeyi fonksiyona gönder
ortalamahesapla(liste2)
