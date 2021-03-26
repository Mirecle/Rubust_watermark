from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pywt import dwt2, idwt2
import math
import random
import genwatermark
import insert

def rgb2Gray(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return img_gray

def gray2rgb(oimg,grey,ogrey):
    
    for i in range(len(oimg)): 
        for j in range(len(oimg[i])): 
            if (grey[i][j]==ogrey[i][j]).all():
                break
            else:
                delat=ogrey[i][j]-grey[i][j]
                oimg[i][j][0]=oimg[i][j][0]-delat
                oimg[i][j][1]=oimg[i][j][1]-delat
                oimg[i][j][2]=oimg[i][j][2]-delat

    return oimg
def get_size(file):
    # 获取文件大小:KB
    size = os.path.getsize(file)
    return size / 1024

def resize_image(infile,outfile,x_s,y_s):
    """修改图片尺寸
    :param infile: 图片源文件
    :param outfile: 重设尺寸文件保存地址
    :param x_s: 设置的宽度
    :return:
    """
    im = insert.readImage(infile)
    im = im[0:x_s,0:y_s]
    #out = im.resize((x_s, y_s),Image.ANTIALIAS)
    cv2.imwrite(outfile, im,[int(cv2.IMWRITE_JPEG_QUALITY),95])
def readImage(address):
    image=cv2.imread(address)
    b,g,r = cv2.split(image)
    img = cv2.merge([r,g,b])#由于 cv2.imread读取的图片是bgr形式,转换为rgb格式
    return img
def get_outfile(infile, outfile):
    if outfile:
        return outfile
    dir, suffix = os.path.splitext(infile)
    outfile = '{}-out{}'.format(dir, suffix)
    return outfile

def compress_image(infile, outfile='', mb=3072, step=1, quality=40):
    """不改变图片尺寸压缩到指定大小
    :param infile: 压缩源文件
    :param outfile: 压缩文件保存地址
    :param mb: 压缩目标，KB
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件地址，压缩文件大小
    """
    o_size = get_size(infile)
    if o_size <= mb:
        return infile
    outfile = get_outfile(infile, outfile)
    while o_size > mb:
        im = Image.open(infile)
        #出现 OSError: cannot write mode RGBA as JPEG 错误时加入下方语句
        #im = im.convert("RGB")
        im.save(outfile, quality=quality)
        if quality - step < 0:
            break
        quality -= step
        o_size = get_size(outfile)
    return outfile, get_size(outfile)
def checksize(image):
    M,N =image.shape[:2]
    print('M=%d,N=%d'%(M,N))
    if M%128!=0:
        print('high not suit,remain%d'%(M%128))
        m=M-M%128
    else:
        m=M
    if N%128!=0: 
        print('width not suit,%d'%(N%128))
        n=N-N%128
    else:
        n=N
    
    return m,n
# outfile,get_size=compress_image('lena512.jpg','outfile.jpg')
if __name__ == "__main__":
    
    # compress_image('5.jpg','1-out.jpg',1000)
    img=readImage('lena512.jpg')
    #resize_image('22.jpg','22-out.jpg',256, 512)
    print(checksize(img))
    # gray=rgb2Gray(img)
    # reimg=gray2rgb(img,gray,gray)
    # plt.subplot(111)
    # plt.imshow(reimg)
    # plt.show()