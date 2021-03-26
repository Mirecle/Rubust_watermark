#基于DCT 的水印注入程序
#insert copy.py中没有置乱
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
ownership="0811742weiboupload"
Oaddress='yaoming2.jpg'
# Waddress='black32.jpg'
In='10101010110011000011110000101010010111000001101000000001010001110100010010001110011111101101010100111011100010001010101011001100'

# def edit(param,im):
#     if param == 'BOTTOM':
#         #进行上下颠倒
#         out = im.transpose(Image.FLIP_TOP_BOTTOM)
#         #进行左右颠倒
#     if param == 'RIGHT':
#         out =im.transpose(Image.FLIP_LEFT_RIGHT)
#         # 进行旋转90
#     if param == '90':
#         out = im.transpose(Image.ROTATE_90)
#         # 进行旋转180
#     if param == '180':
#         out = im.transpose(Image.ROTATE_180)
#         # 进行旋转270
#     if param == '270':
#         out = im.transpose(Image.ROTATE_270)
#     else:
#         return None
#     return out
#图片读取
def readImage(address):
    image=cv2.imread(address)
    # b,g,r = cv2.split(image)
    # img = cv2.merge([r,g,b])#由于 cv2.imread读取的图片是bgr形式,转换为rgb格式
    return image

def recolor(image):
    b,g,r = cv2.split(image)
    img = cv2.merge([r,g,b])#由于 cv2.imread读取的图片是bgr形式,转换为rgb格式
    return img
#图片转化为灰度图
def rgb2Gray(image):
    b,g,r = cv2.split(image)
    b=b.astype(np.float64)
    g=g.astype(np.float64)
    return b,g,r

def gray2rgb(b,g,r):
    merged = cv2.merge([b,g,r])
    return merged
#图片展示
def showImage(image):
    plt.imshow(image,'gray')
    plt.axis("off")
    #plt.show()
#图片二值化
def thresh_binary(image):
    ret,thresh=cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    return thresh

# showImage(gray)

#图片切割
def cut(Oimage,K_w,K_h):
    m,n=Oimage.shape
    width=int(m/K_w)
    high=int(n/K_h)
    subImage=[]
    for i in range(0,width):
        subline=[]
        for j in range(0,high):
            sub=Oimage[i*K_w:(i+1)*K_w,j*K_h:(j+1)*K_h]
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
    cA,(cH,cV,cD)=dwt2(Oimage,'haar')
    M,m=cA.shape #512 原图的长和宽
    N,n=Wimage.shape #32 水印的长和宽
 
    K_w=int(M/N) #分割倍数
    K_h=int(m/n)
    cutImage=cut(cA,K_w,K_h)
    final_image=[]
    Wimage=Arnold(Wimage,1,1,19)
    for i in range(len(cutImage)):
        sub=[]
        for j in range(len(cutImage[i])):
            
            sub_dct = cv2.dct(np.float32(cutImage[i][j]))         #进行离散余弦变换
            n,N=sub_dct.shape
            
            if Wimage[i][j]==0:
#[int(n/2):int(n/2+4),int(n/2-1):int(n/2+3)]
                sub_dct[int(n/2):int(n/2+4),int(n/2-1):int(n/2+3)]=sub_dct[int(n/2):int(n/2+4),int(n/2-1):int(n/2+3)]+alfa*d1
            elif Wimage[i][j]==255:
                sub_dct[int(n/2):int(n/2+4),int(n/2-1):int(n/2+3)]=sub_dct[int(n/2):int(n/2+4),int(n/2-1):int(n/2+3)]+alfa*d2
                # sub_dct[int(n/2):int(n/2+4),int(n/2-1):int(n/2+3)]=sub_dct[int(n/2):int(n/2+4),int(n/2-1):int(n/2+3)]+alfa*d2
            sub_idct = cv2.idct(sub_dct)
            sub.append(sub_idct)
        final_image.append(sub)
        img=stitch(final_image)
    img= idwt2((img,(cH,cV,cD)), 'haar')
    return img

#相关系数计算
def cor(z,x):
    t=0
    t1=0
    t2=0
    for i in range(len(z)):
        t+= z[i]*x[i]
        t1 += z[i]**2
        t2 += x[i]**2
    result = t/(t1**0.5+t2**0.5)
    return result

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
def Accuracy(In,Out):
    total=len(In)
    error=0
    for i in range(total):
        if In[i]!=Out[i]:
            error+=1
    return float((1-error/total)*100)
# 水印提取
def extract(img,d1,d2,N):
    cA,(cH,cV,cD)=dwt2(img,'haar')
    watermark=np.zeros((32,32),dtype='int')
    M,m=cA.shape #256 原图的长和宽
    K_w=int(M/N) #分割倍数
    K_h=int(m/N)
    cutImage=cut(cA,K_w,K_h)
    final_image=[]
    for i in range(len(cutImage)):
        sub=[]
        for j in range(len(cutImage[i])):
            exdct=cv2.dct(np.float32(cutImage[i][j]))
            n,N=exdct.shape
            sub_dct = exdct[int(n/2):int(n/2+4),int(n/2-1):int(n/2+3)]
            sub1=sub_dct.reshape(1, 16)[0]
            d11=d1.reshape(1,16)[0]
            d21=d2.reshape(1,16)[0]

            if cor(sub1,d11)>=cor(sub1,d21):
                watermark[i][j]=0
            else:
                watermark[i][j]=255
    return DeArnold(watermark,1,1,19)


if __name__ == "__main__":
    ownership='aacc3c2a5c1a0147448e7ed53b88aacc'   #前flag：前4位，user_id：8位，SNP_id：2位,后flag：4位
    Oimage=readImage(Oaddress)
    (b,g,image)=rgb2Gray(Oimage)
    Wimage=genwatermark.genWatermark(ownership,32)
    #Wimage=readImage(Waddress)
    #Wimage=rgb2Gray(Wimage)
    #Wimage=thresh_binary(Wimage)

    # showImage(Oimage)
    # showImage(thresh_Wimage)
    # showImage(Oimage)
    # plt.show()
    M,m=image.shape #512 原图的长和宽
    N,n=Wimage.shape #32 水印的长和宽
    K=int(M/N) #分割倍数
    d1=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    d2=np.array([[1,1,1,1],[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1]])
    insert_image=insert(image,Wimage,d1,d2,15)
    
    insert_image = insert_image.astype(np.float64)
 
    backimage=gray2rgb(b,g,insert_image)
    # a=1600
    # test=np.zeros((a,a,3),dtype='float32')
    
    # backimage[0:a,0:a]=test
    # cv2.imwrite("5-out-out.png", backimage)
    cv2.imwrite('yaoming2-out.jpg', backimage,[int(cv2.IMWRITE_JPEG_QUALITY),95])
    eximg=readImage('yaoming2-out.jpg')
    (b,g,extracts)=rgb2Gray(eximg)
    watermark=extract(extracts,d1,d2,N)
    result,result_ownership=genwatermark.deWatermark(watermark,len(genwatermark.encode(ownership)))
    
    print(result)
    accuracy=Accuracy(In,result_ownership)
    print('准确率:%f ' %accuracy)

    # backimage=recolor(backimage)/255

    # Oimage=recolor(Oimage)
   
    # plt.subplot(131)
    # showImage(backimage)
    # plt.subplot(132)
    # showImage(Oimage)
    # plt.subplot(133)
    # showImage(watermark)
    # plt.show()
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