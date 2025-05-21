import turtle

# Ekranı başlat
screen = turtle.Screen()
screen.bgcolor("white")

# Turtle objesi oluştur
t = turtle.Turtle()
t.shape("turtle")
t.speed(10)  # Hızı artırmak için 10 kullanabiliriz

# Daire çizme fonksiyonu
def draw_circle(radius):
    t.circle(radius)

# İç içe daireler çizme
def draw_nested_circles(num_circles, radius_increment):
    for i in range(num_circles):
        draw_circle(10 + i * radius_increment)  # Dairenin çapını artır
        t.penup()  # Çizimden sonra kalemi kaldır
        t.setpos(0, -(10 + i * radius_increment))  # Yeni daireyi merkezde konumlandır
        t.pendown()  # Çizime tekrar başla

# Daireleri çiz
draw_nested_circles(10, 15)

# Ekranda tut
turtle.done()
