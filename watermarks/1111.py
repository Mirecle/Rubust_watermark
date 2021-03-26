import cv2 as cv2
import genwatermark
import insert
import numpy as np
src=insert.readImage('./lena512/lena512-out.jpg')
M,N,Z=src.shape
NoiseImg=np.zeros([M,N,Z]) 
for i in range(M):
    for j in range(N):
        for z in range(Z):
            NoiseImg[i][j][z]=src[i][j][z]