#namelist=[]

#name1=input("lütfen isim gir: ")

#name2=input("lütfen isim gir: ")

#name3=input("lütfen isim gir: ")

#namelist.append(name1)
#namelist.append(name2)
#namelist.append(name3)
#print(namelist)

namelist=[]

kac=int(input("kaç tane isim gireceksin:  "))

while len(namelist) < kac:
    isim=input("lütfen isim giriniz: ")
    namelist.append(isim)
    print(namelist)