from numpy import *
import operator
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = array(tile(inX, (dataSetSize,1)) - dataSet)
    # sqDiffMat = (array(diffMat ** 2))
    sqDistances = (diffMat ** 2).sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}

    for i in range(k):  
        voteIlabel = labels[sortedDistIndicies[i]]
        vote = int(voteIlabel)
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]