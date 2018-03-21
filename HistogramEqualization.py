import cv2
from numpy import *

def HistGraphGray(image,color):
    hist = cv2.calcHist([image],[0],None,[256],[0.0,255.0])
    histGraph = np.zeros([256,256,3],np.uint8)
    m = max(hist)
    hist = hist * 220 / m
    for h in range(256):
        n = int(hist[h])
        cv2.line(histGraph,(h,255),(h,255-n),color)
    return histGraph



def junhenghua(image):
    height,width = shape(image)
    NumPixel = zeros((256))

    for i in range(0,height):
        for j in range(0,width):
            NumPixel[image[i][j]] = NumPixel[image[i][j]]+1


    ProbPixel = zeros((256))
    for i in range(0,256):

        ProbPixel[i] = NumPixel[i]/(height * width * 1.0)

    CumuPixel = zeros((256))
    for i in range(0,256):
        if i == 0 :
            CumuPixel[i] = ProbPixel[i]
        else:
            CumuPixel[i] = CumuPixel[i-1]+ProbPixel[i]


    CumuPixel = uint8(255.* CumuPixel+0.5)

    img2 = zeros((height,width))
    for i in range(0,height):
        for j in range(0,width):
            img2[i][j] = CumuPixel[image[i][j]]

    return img2
    # print shape(img2)
    # print image

    # color = [255, 255, 255]
    # histGraph1 = HistGraphGray(image, color)
    # cv2.imshow("Hist Gray", histGraph1)
    # os.system("pause")
    # new_img2 = Image.fromarray(img2)
    # new_img2.show()

    # cv2.imshow('Hist Gray',image)
    # cv2.waitKey(0)


# if __name__ == '__main__':
#     img = cv2.imread('D:\PyCharm\PyCharmProjects\Face/2.bmp', 0)
#     junhenghua(img)
# # color = [255,255,255]
# # histGraph1 = HistGraphGray(img,color)
# # cv2.imshow("Hist Gray",histGraph1)
#     cv2.waitKey(0)

