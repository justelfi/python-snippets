import os

print(os.name) #işletim sistemi new tech. şuan ki windows

for i in os.environ:
    print(i, os.environ[i])

#print(os.listdir('C:\WINDOWS\system32')) 

for i in os.walk('C:/Users/Damla/Desktop/elif'):
    print(i)

print(os.getcwd()) # burda çalışıyorsun

print(os.chdir('C:/Users/Damla/Desktop/elif')) 

#os.mkdir('C:/Users/Damla/Desktop/elif/yeni2') # dosya oluşturdu

#os.rmdir('C:/Users/Damla/Desktop/elif/yeni2')  #dosyayı sildi(yeni2)

#os.removedirs() [dosyaları kaldırırdikkat etmen lasımm]

#os.removedirs('C:/Users/Damla/Desktop/elif/yeni/yeniin/yeniin2')

#os.rename('C:/Users/Damla/Desktop/elif/yeni2', ('C:/Users/Damla/Desktop/elif/yeni1'))
# ismini değiştirdi ve oluşturdu

#cmd="notepad"
#os.system(cmd)

print(os.path.exists('C:/Users/Damla/Desktop/elif')) # elif adında dosya var True





