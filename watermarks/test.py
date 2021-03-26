import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pywt import dwt2, idwt2

from PIL import Image
import math
import random
import genwatermark
import compress
import insert
import os
def rgb2Gray(image):
    b,g,r = cv2.split(image)
    b=b.astype(np.float64)
    g=g.astype(np.float64)
    return b,g,r


if __name__ == "__main__":
    ownership='aacc3c2a5c1a0147448e7ed53b88aacc'
    In='10101010110011000011110000101010010111000001101000000001010001110100010010001110011111101101010100111011100010001010101011001100'

    Oaddress='/home/ming/Desktop/watermarks/2/2-out-out.jpg'
    d1=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    d2=np.array([[1,1,1,1],[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1]])
    size=os.path.getsize(Oaddress)
    for i in range(1,10):
        print(i)
        rate=i/10
    
        compress.compress_image(Oaddress,Oaddress[:-4]+str(rate)+'1.jpg',size*rate/1000,1,95)
        Oimage=insert.readImage(Oaddress[:-4]+str(rate)+'1.jpg')
        
        (b,g,extracts)=rgb2Gray(Oimage)
        watermark=insert.extract(extracts,d1,d2,32)
        result,result_ownership=genwatermark.deWatermark(watermark,128)
        
        print(result)
        accuracy=insert.Accuracy(In,result_ownership)
        print('压缩率 %f  准确率:%f ' %(rate,accuracy))