#coding:utf-8
from numpy import *
import LBP
import cv2
import knn
import HistogramEqualization
#图片大小
rows = 92
cols = 112
imagesize=cols*rows
#人脸类别数：
ClassNum = 40
#每类脸的训练样本数：
train_samplesize = 5
#总训练样本数：
train_tol = ClassNum*train_samplesize
#
train = range(1,train_samplesize)
#特征维数
Eigen_num = 40
#每类脸的测试样本数：
new_testSampleSize = 5
new_test_tol = new_testSampleSize
#分类器计数变量：
test_NUM = 1
#
# tol_num = 10

# def loadImageSet():
#     FaceMat = mat(zeros((train_tol,imagesize)))
#     label = mat(zeros((1,train_tol)))
#     m=0
#     for i in range(1,ClassNum+1):
#         for j in range(1,train_samplesize+1):
#             try:
#                  img = cv2.imread('‪D:\PyCharm\PyCharmProjects\ORL\s'+str(i)+'_'+str(j)+'.bmp',0)
#             except:
#                 print 'load %s failed'%i
#
#             FaceMat[m,:] = mat(img).flatten()
#             label[:,m] = i
#             m = m + 1
#     return FaceMat,label

def loadImageSet_many(test_NUM_IN):
    FaceMat = mat(zeros((new_test_tol, imagesize)))
    m = 0
    i= test_NUM_IN
    for j in range(1, new_testSampleSize + 1):
        try:
            img = cv2.imread('./ORL/s' + str(i) + '_' + str(j + new_testSampleSize) + '.bmp', 0)
            # img = HistogramEqualization.junhenghua(img)
            print 'loadmany-> ./ORL/s' + str(i) + '_' + str(j + new_testSampleSize)
        except:
            print 'load %s failed' % i
        FaceMat[m, :] = mat(img).flatten()
        m = m + 1
    return FaceMat

def Eigenface(FaceMat,Eigen_num):
    NN,Train_num = shape(FaceMat)
    Mean_Image = mean(FaceMat,1)#参数为1，对各行求均值
    diffTrain = FaceMat - Mean_Image

    eigvals, eigVects = linalg.eig(mat((diffTrain.T * diffTrain)/(Train_num-1)))
    eigSortIndex = argsort(-eigvals)  # 从大到小排序，默认从小到大，参数为负表示降序
    V = mat(eigVects[:, eigSortIndex[:Eigen_num]])#取特征向量的前四十维

    #取前40大的特征值

    for i in range(1,Eigen_num+1):
        disc_value = mat(eigvals[eigSortIndex[:i]])


    disc_set = mat(zeros((NN,Eigen_num)))

    Train_set = diffTrain / sqrt(Train_num - 1)
    for k in range(0, Eigen_num):
        a = Train_set * V[:,k] #a为10000*1
        b =float (1/sqrt(disc_value[:,k]))
        disc_set[:,k] = b*a
    disc_set = real(disc_set)

    return disc_set,disc_value,Mean_Image


def count(disc_set,W_LDA,train_final,label,test_NUM_IN):
    #####准确率统计代码LDA#######
    # newImg = loadImageSet_many()
    print 'test_num is '+str(test_NUM_IN)
    new = loadImageSet_many(test_NUM_IN)
    # newImg = LBP.LBP(rows,cols,new.T).T
    newImg = new
    newImg_pro = disc_set.T*newImg.T
    newImg_final = W_LDA.T*newImg_pro
    i=0
    suM =0
    while(i<new_test_tol):
        Class = knn.classify0(newImg_final.T[i], train_final.T, label.T, 7)
        # print Class
        if(Class==test_NUM_IN):
            suM = suM+1
        i=i+1
    statistic = float(suM)/new_test_tol
    print 'test_NUM_IN='+str(test_NUM_IN)+' '+str(statistic)
    return statistic

# def countPCA(disc_set,train_pro,label):
#     newImg = loadImageSet_many()
#     newImg_pro = disc_set.T*newImg.T
#     i=0
#     suM =0
#     while(i<new_test_tol):
#         Class = knn.classify0(newImg_pro.T[i], train_pro.T, label.T, 7)
#         if(Class==test_NUM):
#             suM = suM+1
#         i=i+1
#     statistic = float(suM)/new_test_tol
#     print statistic
#     return statistic

def LDA(FaceMat,label):
    FaceMat_set = FaceMat.T
    disc_set,disc_value,MeanImage = Eigenface(FaceMat_set,Eigen_num)
    #训练样本的第一次投影
    train_pro = disc_set.T*FaceMat.T
    #训练样本总体均值及每类均值
    total_mean = mean(train_pro,1)
    EachClassMean = mat(zeros((Eigen_num,ClassNum)))
    EachClassNum = mat(zeros((1,ClassNum)))
    m=0
    temp = mat(zeros((Eigen_num,train_samplesize+1)))
    for i in range(1,ClassNum+1):
        for j in range(1,train_samplesize+1):
            temp[:,j]= train_pro[:,m]
            m=m+1
        EachClassMean[:,i-1] = mean(temp,1)

    #构造Fai_b,Fai_w,计算Sb，Sw

    Fai_b = mat(zeros((Eigen_num,ClassNum)))#每一类
    Fai_w = mat(zeros((Eigen_num,train_tol)))#总类

    #计算类间差
    temp1 = EachClassMean - total_mean
    Fai_b = sqrt(train_samplesize)*temp1

    #计算类内差
    for i in range(0,train_tol):
        Fai_w[:,i] = train_pro[:,i] - EachClassMean[:,int(label[0,i])-1]

    Sb = Fai_b*Fai_b.T
    Sw = Fai_w*Fai_w.T

    LDA_dim = ClassNum-1

    eig_val,eig_vec = linalg.eig(Sw.I*Sb)
    eigSortIndex = argsort(-eig_val)  # 从大到小排序，默认从小到大，参数为负表示降序
    W_LDA = mat(eig_vec[:, eigSortIndex[:LDA_dim]])  # 取LDA方向
    a = 0
    #训练样本再次投影
    train_final = W_LDA.T*train_pro

    for test_NUM in range(1,ClassNum+1):
        print 'test_num in for'+str(test_NUM)
        a += (count(disc_set,W_LDA,train_final,label,test_NUM));
    print 'average is '+str(float(a/ClassNum))