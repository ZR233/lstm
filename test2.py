import numpy as np
from PIL import Image


im = Image.open("cath.jpg")
print(im.mode)
print(im.size)
w,h = im.size
size = 100
print(w,h)
if (w <= h):
    p = float(size)/w*h
    im = im.resize((size, int(p)))
    w,h = im.size
    x = (h-size)/2
    im = im.crop((0,x,100,100+x))
else:
    p = float(size)/h*w
    im = im.resize((int(p), size))
    im.save('cat2.png')
    w,h = im.size
    x = int((w-size)/2)
    im = im.crop((x,0,100+x,100))
print(im.size)
resized = np.array(im)
print(resized.shape)
im.save('cat3.png')

batch_y = [[0]]