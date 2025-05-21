import turtle
a=turtle.Turtle(shape="triangle")
a.speed(1)
arkaplan=turtle.getscreen()
arkaplan.bgcolor("pink")
a.color("white")

#for i in range(1,60):
 #a.circle(200)
 #a.left(20)

def kareyap(uzunluk):
   
    for i in range(4):
        a.forward(uzunluk)
        a.left(90)
uzunlugumuz=int(input("kenar uznlunğu gir: "))
kareyap(uzunlugumuz)