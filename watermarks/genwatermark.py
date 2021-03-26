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


def showImage(image):
    plt.imshow(image,'gray')
    plt.axis("off")
    #plt.show()
#########################归属权信息二进制字符串生成
def encode(s):

    bins=bin(int(s,16))[2:] 
    return bins
##########################二进制信息提取归属权信息
def decode(s):
    ownership=hex(int(s,2))[2:]

    return ownership

##################水印生成
def genWatermark(ownership,N):
    ownership=encode(ownership) #把归属信息转化为2进制
    
    lengh=len(ownership) #lengh保存归属信息的二进制长度，实际为128
    ownership=list(ownership)#将其从字符串转化为list形式
    height=int(lengh/N)#水印要求为N*N格式，则lengh长的信息能铺满height行，实际为128/32=4行
    ownership=np.array(ownership).reshape(height,N)#将list转换为32*4的矩阵
    watermark=np.zeros((N,N),dtype='int')#初始化32*32空白的水印矩阵
    for i in range(N):
        for j in range(N):
            if ownership[i%height][j] =='0':
                watermark[i][j]=0
            elif ownership[i%height][j] =='1':
                watermark[i][j]=255
        
    return watermark
    #for i in N:
###################水印解码
def deWatermark(watermark,lengh):
    N,n=watermark.shape
    #watermark=watermark/255 #归一化为0和1
    ownership=''
    round=[]
    height=int(lengh/N)#一轮归属信息的行数为128/4=4行
    index=int(N/height)# 32/4=8轮
    if N/height-index!=0:
        print('归属信息长度和水印矩阵长宽不匹配!')
        exit()

    for i in range(index):
        round.append(watermark[i*height:i*height+height][:])
    #ownership格式为index个列表，每个列表为lengh*N/ 8个列表，每个列表4*32
    for i in range(height):
        for j in range(N):
            sum=0
            for z in range(index):
                sum+=round[z][i][j]
            if sum>=1275:
                ownership=ownership+'1'
            else:
                ownership=ownership+'0'
    return decode(ownership),ownership
# encode_chain=encode(ownership)
# print(len(encode_chain))
# decode_chain=(decode(encode_chain))
# print(decode_chain)
#print(hex.decode('utf-8'))
if __name__ == "__main__":
    ownership='aacc3c2a5c1a0147448e7ed53b88aacc'   #前flag：前4位，user_id：8位，SNP_id：2位,后flag：4位
    ownership2='ca'
    Oaddress='lena512.jpg'
    Waddress='black32.jpg'
    watermark=genWatermark(ownership,32)
    showImage(watermark)
    plt.show()
    result=deWatermark(watermark,len(encode(ownership)))
    print(result)
    #print(genWatermark(ownership,32)[0])