import random

def rastgele_sayi_sec():
    return random.sample(range(10), 4)  # 10 farklı rakamdan 4 tane seçilir

def tahmin_kontrolu(gercek_sayi, tahmin):
    sonuc = []
    for i in range(4):
        if tahmin[i] == gercek_sayi[i]:  # Hem rakam hem yer doğru
            sonuc.append("D")
        elif tahmin[i] in gercek_sayi:  # Rakam doğru ama yer yanlış
            sonuc.append("Y")
        else:
            sonuc.append("-")  # Yanlış

    # Her rakam için D ve Y'yi döndürür
    return "".join(sonuc)

def kullanici_tahmini_al(gercek_sayi):
    while True:
        try:
            tahmin = input(f"4 farklı rakam girin (örn: 1234) veya 'HELP' yazın: ").strip().upper()

            # Eğer kullanıcı HELP yazarsa doğru sayıyı göster
            if tahmin == "HELP":
                print(f"🔎 Doğru Sayı: {''.join(map(str, gercek_sayi))}")
                continue

            # Sayı kontrolü: Girilen değer tamamen rakam olmalı
            if not tahmin.isdigit():
                raise ValueError("⚠️ Sadece rakam girilmelidir.")

            # Uzunluk kontrolü: 4 rakam girilmeli
            if len(tahmin) != 4:
                raise ValueError("⚠️ 4 rakam girilmelidir.")

            # Tekrar eden rakam kontrolü
            if len(set(tahmin)) != 4:
                raise ValueError("⚠️ Her rakam farklı olmalıdır.")
            
            return [int(x) for x in tahmin]

        except ValueError as e:
            print(f"Hata: {e}. Lütfen tekrar deneyin.")

# Oyunu başlat
print("🎯 10 tahmin hakkınız var. Rakamları tahmin etmeye çalışın!")

# Başlangıçta doğru sayıyı belirler
gercek_sayi = rastgele_sayi_sec()
tahmin_hakki = 10  # Toplam 10 tahmin hakkı

while True:
    while tahmin_hakki > 0:
        # Son 3 hak kaldığında kullanıcıyı uyar
        if tahmin_hakki == 3:
            print("⚠️ Son 3 hakkınız kaldı! HELP yazabilirsiniz!")

        tahmin = kullanici_tahmini_al(gercek_sayi)
        
        # Eğer HELP yazıldıysa, tekrar giriş istemesi için döngüye devam et
        if tahmin is None:
            continue

        sonuc = tahmin_kontrolu(gercek_sayi, tahmin)
        
        print(f"📌 Sonuç: {''.join(map(str, tahmin))} -> {sonuc}")

        tahmin_hakki -= 1  # Her tahminde hakkı azalt
        print(f"⚡ Kalan tahmin hakkınız: {tahmin_hakki}")

        if tahmin == gercek_sayi:
            print("🎉 Tebrikler! Doğru tahmin ettiniz.")
            break  # Doğru tahmin yapıldığında çıkıyoruz

    if tahmin_hakki == 0:
        print(f"😢 Tahmin hakkınız bitti! Doğru sayı: {''.join(map(str, gercek_sayi))}")
    
    # Yeni bir oyun başlatmak için doğru sayıyı yenilemeliyiz
    devam_et = input("Yeni bir oyun oynamak ister misiniz? (E/H): ").strip().upper()
    if devam_et == "E":
        # Yeni bir doğru sayı oluşturuyoruz ve tahmin hakkını sıfırlıyoruz
        gercek_sayi = rastgele_sayi_sec()
        tahmin_hakki = 10  # Her yeni oyun başladığında tahmin hakkı sıfırlanmalı
    else:
        print("Oyunu bitiriyorsunuz. Teşekkürler!")
        break
