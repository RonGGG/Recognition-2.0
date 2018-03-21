import cv2
from numpy import *
import os
from PIL import Image
import HistogramEqualization
def loadImage(add,ClassNum,countInSameClass,image_total,sizeOfImage):
    FaceMat = mat(zeros((image_total, sizeOfImage)))
    label = mat(zeros((1, image_total)))
    m=0
    for i in range(1,ClassNum+1):
        for j in range(1,countInSameClass+1):
            try:
                img = cv2.imread(add+str(i)+'_'+str(j)+'.bmp',0)
                cv2.imshow('1', img)
                img = HistogramEqualization.junhenghua(img)
                Image.fromarray(img).show()
                cv2.waitKey(0)
                os.system("pause")
            except:
                print 'load %s failed'%i
            FaceMat[m,:] = mat(img).flatten()
            label[:,m] = i
            m = m + 1
    return FaceMat.T,label