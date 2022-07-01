'''
Description: 
Author: xiaoer
Date: 2022-06-27 12:10:50
LastEditTime: 2022-07-01 10:32:22
LastEditors:  
'''
from numpy import *
import numpy 


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open(r'logistic\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        #常数项+参数项
        labelMat.append(int(lineArr[2]))
        #数据集label
    return dataMat,labelMat


def sigmoid(inX):
    #一个神奇的函数
    return 1.0/(1+exp(-inX))
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    #转置
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    #步长       
    maxCycles = 500
    #迭代次数
    weights = ones((n,1))
    #系数矩阵
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
        #按照差值方向调整回归系数
    return weights

def stocGradAscent0(dataMatrix,classLabels):
    #随机梯度上升
    m,n=shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*numpy.array(weights)))
        error = classLabels[i]-h
        weights = weights+alpha*error*numpy.array(dataMatrix[i])
    return weights
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    #改进的随机梯度上升
    m,n=shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+i+j)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            #随机选择更新
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights+error*alpha*numpy.array(dataMatrix[randIndex])
            del(dataIndex[randIndex])
    return weights
def plotBestFit(weights):
    print(weights)
    #绘图
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    weights = array(weights)
    n = shape(dataMat)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    # 通过指定开始值、终值和步长来创建表⽰等差数列的⼀维数组，返回给定间隔内的均匀间隔值，注意得到的结果数组不包含终值。
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    # print(x)
    # print(y)
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if(prob>0.5):return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('logistic\horseColicTraining.txt')
    frTest = open('logistic\horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,600)
    errorCount = 0.0
    numTestVect = 0.0
    for line in frTest.readlines():
        numTestVect+=1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount+=1
    errorRate = errorCount/numTestVect
    print(f'错误率是：{errorRate}')
    return errorRate
colicTest()