import cv2 as cv2
import genwatermark
import insert
import numpy as np
import random

from matplotlib import pyplot as plt  

def PepperandSalt(src,percetage):
   
    M,N,Z=src.shape
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.randint(0,1)<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

def Cropping(src,percetage):
   
    M,N,Z=src.shape
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.randint(0,1)<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg
# salt=np.random.random(image.shape)>.988

# pepper=np.random.random(image.shape)>.988


# image=np.maximum(salt*170,image)
# image=np.minimum(pepper*30+image*(~pepper),image)
image=insert.readImage('./lena512/lena512-out.jpg')

aaa=PepperandSalt(image,0.008)

d1=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
d2=np.array([[1,1,1,1],[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1]])
# img = cv.imread(r'./2/2-out-out.jpg', 1)
# rows, cols, channels = img.shape
# rotate = cv.getRotationMatrix2D((rows*0.5, cols*0.5), 360, 1)
# '''
# 第一个参数：旋转中心点
# 第二个参数：旋转角度
# 第三个参数：缩放比例
# '''
ownership='aacc3c2a5c1a0147448e7ed53b88aacc'

# Waddress='black32.jpg'
In='10101010110011000011110000101010010111000001101000000001010001110100010010001110011111101101010100111011100010001010101011001100'

# res = cv.warpAffine(image, rotate, (cols, rows))
(b,g,extracts)=insert.rgb2Gray(aaa)
watermark=insert.extract(extracts,d1,d2,32)
result,result_ownership=genwatermark.deWatermark(watermark,len(genwatermark.encode(ownership)))
    
print(result)
accuracy=insert.Accuracy(In,result_ownership)
print('准确率:%f ' %accuracy)
plt.subplot(131)
plt.imshow(image)
plt.title('original image\n' ,fontdict={'weight':'normal','size': 10})
plt.xticks([]), plt.yticks([])
plt.subplot(132)
plt.imshow(aaa)
plt.title('pepper attack: \n' ,fontdict={'weight':'normal','size': 10})
plt.xlabel('Accuracy:%02f '%accuracy ,y=-1,fontdict={'weight':'normal','size': 10})
plt.xticks([]), plt.yticks([])
 
plt.subplot(133)
plt.imshow(image)
plt.title('original image\n' ,fontdict={'weight':'normal','size': 10})
plt.xticks([]), plt.yticks([])
 
plt.show()