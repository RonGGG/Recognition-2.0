# Recognition-2.0

（一）代码（.py文件）介绍

1）Main_Func:程序入口，调用LoadImage.loadImage(), LBP.LBP(), LDA.LDA

2）LoadImage.py：加载图片文件，函数调用方式是 
“转置后的Facemat，标签label = LoadImage.loadImage(图片路径,人脸类别数目,每一类脸训练数量,总共训练数量,图片大小)”

3）HistogramEqualization.py：均衡化代码文件，函数调用方式是 
“图片 = HistogramEqualization.junhenghua(图片)”函数即可

4）LBP.py：LBP算法文件，函数调用方式：
"转置的FaceMat = LBP.LBP(图片row,图片col,转置的FaceMat, R=2, P=8)"

5）LDA.py：PCA+LDA算法文件，函数调用方式：
"LDA.LDA(转置矩阵FaceMat,标签数组label)"

（二）数据文件（图片库）介绍

1）face：自己拍的照片，共11类，每类500张照片，图片大小100*100，彩图
2）ORL：ORL的数据库，共40类，每类10张照片，图片大小92*112，灰度图
