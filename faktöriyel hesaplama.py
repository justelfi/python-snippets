
import tkinter as tk
from tkinter import messagebox

# Faktöriyel hesaplama fonksiyonu
def faktoriyel(n):
    sonuc = 1
    for i in range(1, n + 1):
        sonuc *= i
    return sonuc

# Hesapla fonksiyonu
def hesapla():
    try:
        sayi = int(entry_sayi.get())
        if sayi < 0:
            raise ValueError("Lütfen negatif olmayan bir sayı girin.")

        # Faktöriyel hesapla
        result = faktoriyel(sayi)

        # Sonucu göster
        label_sonuc.config(text=f"{sayi}! = {result}")

    except ValueError as e:
        messagebox.showerror("Hata", str(e))

# Pencereyi oluştur
root = tk.Tk()
root.title("Faktöriyel Hesaplama")

# Arka plan rengini açık pembe yap
root.config(bg="#FFB6C1")

# Başlık etiketi
label_baslik = tk.Label(root, text="Faktöriyel Hesaplama", font=("Arial", 20, "bold"), fg="#C71585", bg="#FFB6C1")
label_baslik.pack(pady=20)

# Kullanıcıdan sayı girmesini iste
label_sayi = tk.Label(root, text="Bir sayı girin:", font=("Arial", 14), fg="#C71585", bg="#FFB6C1")
label_sayi.pack()

entry_sayi = tk.Entry(root, font=("Arial", 14), width=10)
entry_sayi.pack(pady=10)

# Hesapla butonu
button_hesapla = tk.Button(root, text="Hesapla", font=("Arial", 14), command=hesapla, bg="#32CD32", fg="white")
button_hesapla.pack(pady=10)

# Sonuç etiketini oluştur
label_sonuc = tk.Label(root, text="", font=("Arial", 14), fg="#C71585", bg="#FFB6C1")
label_sonuc.pack(pady=20)

# Pencereyi başlat
root.mainloop()






