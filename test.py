 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import random
Oaddress='lena512.jpg'
Waddress='CCUT64.jpg'
time=2
Wimage=np.zeros((4,4),dtype=int)
zz=0
M,m=Wimage.shape
AN = np.zeros([M,M])
a=1
b=1
for i in range(4):
    for j in range(4):
        Wimage[i][j]=zz
        zz+=1
print(Wimage)

AN=Wimage
print(AN)