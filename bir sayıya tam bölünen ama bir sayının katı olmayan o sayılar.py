
listem=[]

x=int(input("lütfen 1. değeri giriniz: "))
y=int(input("lütfen 2. değeri giriniz: "))

for i in range(x,y):
    
    if i%7 == 0 and i%3 != 0:
        listem.append(str(i))

print(','.join(listem))
