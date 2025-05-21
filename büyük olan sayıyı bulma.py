x=int(input("lütfen 1. değeri gir: "))

y=int(input("lütfen 2. değeri gir: "))

if x>y or y<x:
    print("büyük olan sayi", x)

elif y>x or x<y:
    print("büyük olan sayi", y)

else:
    print("iki sayi birbirine eşittir.")