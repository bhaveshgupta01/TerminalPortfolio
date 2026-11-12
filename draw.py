from turtle import Turtle, done

t = Turtle()
colors = ["red", "orange", "yellow", "green", "blue", "purple"]

for i in range(100):
    t.color(colors[i % 6])
    t.forward(i*10)
    t.right(90)

done()
