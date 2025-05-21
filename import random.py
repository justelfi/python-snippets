import random
a=random.random()
print("random sonuc: ", a)


b=random.uniform(1,22)
print("uniforö sonuc:", b)


c=random.randint(1,8)
print("randint sonuc:", c)



d=random.randrange(1,2)
print("randrange sonuc:", d)


liste=["beyaz", "sari", "lacivert","kirmizi" ]
e=random.choice(liste)
print("secim sonucu:",e)


liste2=["fb","gs", "ts", "bjk"]
random.shuffle(liste2)
print(liste2)
 # import random as rnd (kısalttık)