import random

def rastgele_sayi_sec():
    return random.sample(range(10), 4)  # 10 rakam içinden 4 farklı rakam seçilir

def tahmin_kontrolu(gercek_sayi, tahmin):
    sonuc = []
    for i in range(4):
        if tahmin[i] == gercek_sayi[i]:  # Hem sayı hem yer doğru
            sonuc.append("D")
        elif tahmin[i] in gercek_sayi:  # Sadece sayı doğru, yer yanlış
            sonuc.append("Y")
    return "".join(sonuc)

def kullanici_tahmini_al():
    while True:
        try:
            tahmin = input("4 farklı rakam girin (örn: 1234) veya 'HELP' yazın: ").strip().upper()

            # Eğer kullanıcı HELP yazarsa doğru sayıyı göster
            if tahmin == "HELP":
                print(f"Doğru Sayı: {''.join(map(str, gercek_sayi))}")
                continue

            # Sayı kontrolü: Girilen değer tamamen rakam olmalı
            if not tahmin.isdigit():
                raise ValueError("Sadece rakam girilmelidir.")

            # Uzunluk kontrolü: 4 rakam girilmeli
            if len(tahmin) != 4:
                raise ValueError("4 rakam girilmelidir.")

            # Tekrar eden rakam kontrolü
            if len(set(tahmin)) != 4:
                raise ValueError("Her rakam farklı olmalıdır.")
            
            # Listeye dönüştür
            return [int(x) for x in tahmin]

        except ValueError as e:
            print(f"Hata: {e}. Lütfen tekrar deneyin.")

# Oyunu başlat
print("Rakamları tahmin etmeye çalışın! (4 farklı rakam)")
gercek_sayi = rastgele_sayi_sec()

while True:
    tahmin = kullanici_tahmini_al()
    
    # Eğer HELP yazıldıysa, tekrar giriş istemesi için döngüye devam et
    if tahmin is None:
        continue

    sonuc = tahmin_kontrolu(gercek_sayi, tahmin)
    
    if sonuc:
        print(f"Sonuç: {sonuc}")
    else:
        print("Hiçbir eşleşme yok.")

    if tahmin == gercek_sayi:
        print("Tebrikler! Doğru tahmin ettiniz.")
        break
