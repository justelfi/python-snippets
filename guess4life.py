import random
import tkinter as tk
from tkinter import messagebox

# 4 basamaktan oluşan rastgele bir sayı üretir
def rastgele_sayi_sec():
    return random.sample(range(10), 4)

# Kullanıcının tahminini kontrol eden fonksiyon
def tahmin_kontrolu(gercek_sayi, tahmin):
    sonuc = []
    for i in range(4):
        if tahmin[i] == gercek_sayi[i]:  # Hem rakam hem yer doğru
            sonuc.append("D")
        elif tahmin[i] in gercek_sayi:  # Rakam doğru ama yer yanlış
            sonuc.append("Y")
        else:
            sonuc.append("-")  # Yanlış
    return "".join(sonuc)

# Tahmin yapmayı sağlayan fonksiyon
def tahmin_yap():
    global tahmin_hakki, gercek_sayi

    tahmin = entry_tahmin.get()
    
    # Eğer kullanıcı HELP yazarsa doğru sayıyı göster
    if tahmin.upper() == "HELP":
        messagebox.showinfo("Doğru Sayı", f"Doğru Sayı: {''.join(map(str, gercek_sayi))}")
        return

    # Giriş kontrolü: Girilen değer 4 rakamlı olmalı
    if not tahmin.isdigit() or len(tahmin) != 4 or len(set(tahmin)) != 4:
        messagebox.showerror("Hata", "⚠️ 4 farklı rakam girilmeli!")
        return
    
    tahmin = [int(x) for x in tahmin]  # Girilen tahmini listeye dönüştür

    # Tahmini kontrol et
    sonuc = tahmin_kontrolu(gercek_sayi, tahmin)
    
    # Sonucu ekrana yaz
    label_sonuc.config(text=f"Sonuç: {sonuc}")
    tahmin_hakki -= 1

    # Tahmin hakkını güncelle
    label_hakki.config(text=f"Kalan Tahmin Hakkınız: {tahmin_hakki}")

    # Doğru tahmin yapıldıysa oyunu bitir
    if tahmin == gercek_sayi:
        messagebox.showinfo("Tebrikler", "🎉 Doğru tahmin ettiniz!")
        yeni_oyun()

    # Eğer haklar biterse, doğru sayıyı göster
    if tahmin_hakki == 0:
        messagebox.showinfo("Bitti", f"😢 Başaramadın canımsu! Doğru sayı: {''.join(map(str, gercek_sayi))}")
        yeni_oyun()

# Yeni oyun başlat
def yeni_oyun():
    global tahmin_hakki, gercek_sayi
    gercek_sayi = rastgele_sayi_sec()
    tahmin_hakki = 10  # 10 tahmin hakkı
    label_hakki.config(text=f"Kalan Tahmin Hakkınız: {tahmin_hakki}")
    label_sonuc.config(text="")

# Pencereyi oluştur
root = tk.Tk()
root.title("Rakam Tahmin Oyunu")
root.geometry("500x400")
root.config(bg="#f0f8ff")  # Açık mavi arka plan

# Oyun başlat
gercek_sayi = rastgele_sayi_sec()
tahmin_hakki = 10

# Başlangıç mesajı
label_baslangic = tk.Label(root, text="🎯 10 tahmin hakkınız var. Rakamları tahmin etmeye çalışın!", font=("Arial", 14, "bold"), fg="#ff4500", bg="#f0f8ff")
label_baslangic.pack(pady=10)

# Kullanıcıdan tahmin almak için bir giriş kutusu
label_tahmin = tk.Label(root, text="4 farklı rakam girin (örn: 1234):", font=("Arial", 12, "bold"), fg="#ff6347", bg="#f0f8ff")
label_tahmin.pack(pady=5)

entry_tahmin = tk.Entry(root, font=("Arial", 14), width=20, justify="center", bd=3, relief="sunken")
entry_tahmin.pack(pady=5)

# Tahmin yapmayı sağlayan buton
button_tahmin = tk.Button(root, text="Hadi!!Tahmin Et", font=("Arial", 14), command=tahmin_yap, bg="#32cd32", fg="white", relief="raised", width=15)
button_tahmin.pack(pady=10)

# Buton üzerine gelince renk değişimi
button_tahmin.bind("<Enter>", lambda event: button_tahmin.config(bg="#228b22"))
button_tahmin.bind("<Leave>", lambda event: button_tahmin.config(bg="#32cd32"))

# Kalan tahmin hakkını gösteren etiket
label_hakki = tk.Label(root, text=f"Kalan Tahmin Hakkınız: {tahmin_hakki}", font=("Arial", 14), fg="#ff6347", bg="#f0f8ff")
label_hakki.pack(pady=5)

# Sonuç etiketini göster
label_sonuc = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="#ff4500", bg="#f0f8ff")
label_sonuc.pack(pady=5)

# Yeni oyun başlatma butonu
button_yeni_oyun = tk.Button(root, text="Yeni Oyun Başlat", font=("Arial", 14), command=yeni_oyun, bg="#ff6347", fg="white", relief="raised", width=15)
button_yeni_oyun.pack(pady=10)

# Buton üzerine gelince renk değişimi
button_yeni_oyun.bind("<Enter>", lambda event: button_yeni_oyun.config(bg="#dc143c"))
button_yeni_oyun.bind("<Leave>", lambda event: button_yeni_oyun.config(bg="#ff6347"))

# Enter tuşu ile tahmin yapma
root.bind("<Return>", lambda event: tahmin_yap())

# Pencereyi çalıştır
root.mainloop()
