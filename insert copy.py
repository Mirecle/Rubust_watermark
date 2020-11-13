#二维图像的离散变余弦换（DCT）
#Python3.5
#库：cv2+numpy+matplotlib
#作者：James_Ray_Murphy
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import random
Oaddress='lena512.jpg'
Waddress='CCUT64.jpg'

#图片读取
def readImage(address):
    image=cv2.imread(address)
    b,g,r = cv2.split(image)
    img = cv2.merge([r,g,b])#由于 cv2.imread读取的图片是bgr形式,转换为rgb格式
    return img
#图片转化为灰度图
def rgb2Gray(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return img_gray
#图片展示
def showImage(image):
    plt.imshow(image,'gray')
    plt.axis("off")
    #plt.show()
#图片二值化
def thresh_binary(image):
    ret,thresh=cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    return thresh
image=readImage("yaoming.jpg")
gray=rgb2Gray(image)
# showImage(gray)

#图片切割
def cut(Oimage,K):
    m,n=Oimage.shape
    width=int(m/K)
    high=int(n/K)
    subImage=[]
    for i in range(0,width):
        subline=[]
        for j in range(0,high):
            sub=Oimage[i*K:(i+1)*K,j*K:(j+1)*K]
            subline.append(sub)
        subImage.append(subline)
    return subImage


#拼接图像
def stitch(cut_image):
    i=len(cut_image)
    j=len(cut_image[0])
    height=[]
    for x in range(i):
        tmp=[]
        for y in range(j):
            tmp.append(cut_image[x][y])
        img=np.concatenate(tmp,axis=1)
        height.append(img)
    img3=np.vstack(height)
    return img3

#水印嵌入    
def insert(Oimage,Wimage,d1,d2,alfa):
    M,m=Oimage.shape #512 原图的长和宽
    N,n=Wimage.shape #64 水印的长和宽
    K=int(M/N) #分割倍数
    cutImage=cut(Oimage,K)
    final_image=[]
    for i in range(len(cutImage)):
        sub=[]
        for j in range(len(cutImage[i])):
            sub_dct = cv2.dct(np.float32(cutImage[i][j]))         #进行离散余弦变换
            if Wimage[i][j]==0:
                sub_dct[2:6,2:6]=sub_dct[2:6,2:6]+alfa*d1
            elif Wimage[i][j]==255:
                sub_dct[2:6,2:6]=sub_dct[2:6,2:6]+alfa*d2
            sub_idct = cv2.idct(sub_dct)
            sub.append(sub_idct)
        final_image.append(sub)
        img=stitch(final_image)
    return img

#相关系数计算
def cor(z,x):
    Vx=np.var(x)
    Vz=np.var(z)
    tmpxz=math.sqrt(Vx*Vz)
    return (np.cov(z,x)[0][1])/tmpxz

def Arnold(Wimage,a,b,time):   
    M,m=Wimage.shape
    AN = np.zeros([M,M])
    for i in range(time):
        for y in range(M):
            for x in range(m):
                xx=(x+b*y)%M
                yy=((a*x)+(a*b+1)*y)%M
                AN[yy][xx]=Wimage[y][x]                
        Wimage=AN.copy()
    return Wimage

def DeArnold(Wimage,a,b,time):    
    M,m=Wimage.shape
    AN = np.zeros([M,M])
    for i in range(time):
        for y in range (M):
            for x in range(m):
                xx=((a*b+1)*x-b*y)%M
                yy=(-a*x+y)%M
                AN[yy][xx]=Wimage[y][x]

                
        Wimage=AN.copy()
    return Wimage
# 水印提取
def extract(img,d1,d2,N):
    watermark=np.zeros((64,64),dtype='int')
    M,m=Oimage.shape #512 原图的长和宽
    K=int(M/N) #分割倍数
    cutImage=cut(img,K)
    final_image=[]
    for i in range(len(cutImage)):
        sub=[]
        for j in range(len(cutImage[i])):
            sub_dct = cv2.dct(np.float32(cutImage[i][j]))[2:6,2:6]
            sub1=sub_dct.reshape(1, 16)
            d11=d1.reshape(1,16)
            d21=d2.reshape(1,16)
            if cor(sub1,d11)>=cor(sub1,d21):
                watermark[i][j]=0
            else:
                watermark[i][j]=255
    return watermark
Oimage=readImage(Oaddress)
Oimage=rgb2Gray(Oimage)
Wimage=readImage(Waddress)
Wimage=rgb2Gray(Wimage)
thresh_Wimage=thresh_binary(Wimage)

# showImage(Oimage)
# showImage(thresh_Wimage)
# showImage(Oimage)
# plt.show()
M,m=Oimage.shape #512 原图的长和宽
N,n=Wimage.shape #64 水印的长和宽
K=int(M/N) #分割倍数
d1=np.random.rand(4,4)
d2=np.random.rand(4,4)

insert_image=insert(Oimage,Wimage,d1,d2,60)
watermark=extract(insert_image,d1,d2,N)
plt.subplot(121)
showImage(insert_image)
plt.subplot(122)
showImage(watermark)
plt.show()
# s

# def genMatrix(img,N):
#     i,j=img.shape
#     height=math.ceil(float(i)/N)
#     weighy=math.ceil(float(j)/N)R
#     Matrix=np.zeros((height,weight))
#     for m in range(height):
#         for n in range(weighy):
#             Matrix[i,j]=img[m*N:m*N+N,n*N:n*N+N]
# ####################自写的DCT 太麻烦了，我可懒得搞 
# #C_temp = np.zeros(img.shape)
# #dst = np.zeros(img.shape)
 
 
# #m, n = img.shape
# #N = n
# #C_temp[0, :] = 1 * np.sqrt(1/N)
 
# #for i in range(1, m):
# #     for j in range(n):
#  #         C_temp[i, j] = np.cos(np.pi * i * (2*j+1) / (2 * N )
# #) * np.sqrt(2 / N )
 
# #dst = np.dot(C_temp , img1)
# #dst = np.dot(dst, np.transpose(C_temp))
 
# #dst1= np.log(abs(dst))  #进行log处理
 
# #img_recor = np.dot(np.transpose(C_temp) , dst)
# #img_recor1 = np.dot(img_recor, C_temp)
 
# #自带方法
 
# img_dct = cv2.dct(img1)         #进行离散余弦变换
 
# img_dct_log = np.log(abs(img_dct))  #进行log处理
# img2=img_dct[220:330,220:330]
# i,j=img2.shape
# print(i,j)
# random.seed(key) #根据key产生伪随机数
# psudorandom=random.random()
# for i1 in range(i):
#     for j1 in range(j):
#         if(i1==3 and j1==3):
#             print(img_dct[i1,j1])
# img_recor2 = cv2.idct(img_dct)    #进行离散余弦反变换
 

# #plt.subplot(231)(0, 0, 25
# #plt.imshow(img1, 'gray')
# #plt.title('original image')
# #plt.xticks([]), plt.yticks([])
 
# #plt.subplot(232)
# #plt.imshow(dst1)
# #plt.title('DCT1')
# #plt.xticks([]), plt.yticks([])
 
 
# #plt.subplot(233)
# #plt.imshow(img_recor1, 'gray')
# #plt.title('IDCT1')
# #plt.xticks([]), plt.yticks([])
 
# plt.subplot(131)
# plt.imshow(img, 'gray')
# plt.title('original image')
 
# plt.subplot(132)
# plt.imshow(img_dct_log)
# plt.title('DCT2(cv2_dct)')
 
# plt.subplot(133)
# plt.imshow(img_recor2,'gray')
# plt.title('IDCT2(cv2_idct)')
 
# plt.show()