a=int(input("bir sayı giriniz:"))
b=int(input("bir sayı giriniz:"))

class Mat:
  def __init__(self,fsayi,ssayi) -> None:
      self.fsayi=fsayi
      self.ssayi=ssayi


  def Topla(self):
    return self.fsayi+self.ssayi

  def Carpma(self):
    return self.fsayi*self.ssayi  
  
  def Bolme(self):
    return self.fsayi/self.ssayi

  def Cıkarma(self):
    return self.fsayi-self.ssayi
    
matematik=Mat(a,b)
print(matematik.Topla())