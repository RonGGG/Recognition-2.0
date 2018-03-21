import LoadImage
import LBP
import LDA
ClassNum = 40
countInSameClass = 10
image_total = ClassNum*countInSameClass
sizeOfImage = 112*92
if __name__ == '__main__':
    FaceMat,label = LoadImage.loadImage('./ORL/s',ClassNum,countInSameClass,image_total,sizeOfImage)
    # FaceMat_fromLBP = LBP.LBP(92,112,FaceMat)
    LDA.LDA(FaceMat.T,label)