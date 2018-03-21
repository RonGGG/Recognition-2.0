from numpy import *
import math
def minBinary(pixel):
    length = len(pixel)
    zero = ''
    for i in range(length)[::-1]:
        if pixel[i] == '0':
            pixel = pixel[:i]
            zero += '0'
        else:
            return zero + pixel
    if len(pixel) == 0:
        return '0'



def LBP(rowLBP,colLBP,FaceMat, R=2, P=8):
    Region8_x = [-1, 0, 1, 1, 1, 0, -1, -1]
    Region8_y = [-1, -1, -1, 0, 1, 1, 1, 0]
    pi = math.pi
    LBPoperator = mat(zeros(shape(FaceMat)))
    for i in range(shape(FaceMat)[1]):
        face = array(FaceMat[:, i]).reshape(rowLBP,colLBP)
        W, H = shape(face)
        tempface = mat(zeros((W, H)))
        for x in xrange(R, W - R):
            for y in xrange(R, H - R):
                repixel = ''
                pixel = int(face[x, y])

                for p in [2, 1, 0, 7, 6, 5, 4, 3]:
                    p = float(p)
                    xp = int(x + R * cos(2 * pi * (p / P)))
                    yp = int(y - R * sin(2 * pi * (p / P)))
                    if face[xp, yp] > pixel:
                        repixel += '1'
                    else:
                        repixel += '0'

                tempface[x, y] = int(minBinary(repixel), base=2)
        LBPoperator[:, i] = tempface.flatten().T
        # cv2.imwrite(str(i)+'hh.jpg',array(tempface,uint8))
    return LBPoperator
