from tkinter import *
from PIL import Image, ImageTk, ImageDraw

img_size = (512, 512)

mask = Image.new("1", size=img_size)
draw = ImageDraw.Draw(mask)
DIAMETER = 100
assert DIAMETER % 2 == 0
RADIUS = DIAMETER // 2


def paint(event):
    WHITE = "#FFFFFF"
    x1, y1 = event.x - RADIUS, event.y - RADIUS
    x2, y2 = event.x + RADIUS, event.y + RADIUS
    xy = (x1, y1, x2, y2)
    w.create_oval(*xy, fill=WHITE, outline=WHITE)
    draw.ellipse(xy, fill=1)


master = Tk()
w = Canvas(master, width=img_size[1], height=img_size[0])
w.grid()
w.bind("<B1-Motion>", paint)
img = ImageTk.PhotoImage(Image.open("img.png"))
w.create_image(0, 0, anchor=NW, image=img)

mainloop()

mask.save("mask.png")
