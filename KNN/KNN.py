from os import listdir
import os
from numpy import *
import operator

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels =['A','A','B','B']
    return group, labels
def classify0(index,dataSet,labels,k):
    # shape存储数组的行数和列数
    dataSetSize=dataSet.shape[0]
    # 进行距离计算根号下a^2-b^2
    # 把index复制datasetsize份 然后减去对应每行的dataset 获取他们的差值,
    # tile(A,(a,b)) 把A在x，y上重复a,b次
    diffmat=tile(index,(dataSetSize,1))-dataSet
    sqdiffmat=diffmat**2
    # axis参数为1对行求和，为0对列求和
    sqdistances = sqdiffmat.sum(axis=1)
    distance=sqdistances**0.5
    # 返回从小到大排序的索引值
    sortdistance = distance.argsort()
    classCount={}
    # 选取与当前点距离最近的k个值
    for i in range(k):
        # 统计k个值中不同标签的频率
        votelabel =labels[sortdistance[i]]
        # get()如果有对应key的value返回value 否则返回0
        classCount[votelabel]=classCount.get(votelabel,0)+1
    sortedclasscount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedclasscount)
    return sortedclasscount[0][0]

def file2matrix(filename):
    file=open(filename)
    arrayoflines=file.readlines()
    numberoflines=len(arrayoflines)
    # 生成numberoflines行3列的全零矩阵
    returnmat =zeros((numberoflines,3))
    classLabelVetor=[]
    index=0
    for line in arrayoflines:
        line =line.strip()
        listformat=line.split('\t')
        returnmat[index,:]=listformat[0:3]
        classLabelVetor.append(int(listformat[-1]))
        index +=1
    return returnmat,classLabelVetor


# 数值归一化 newvalue=(oldvalue-min)/(max-min)
def autoNorm(dataset):
    # 对每一行都取到最小值
    minv=dataset.min(0)
    maxv=dataset.max(0)
    ranges =maxv-minv
    normaldataset=zeros(shape(dataset))
    m=dataset.shape[0]
    normaldataset=dataset-tile(minv,(m,1))
    normaldataset=normaldataset/tile(maxv,(m,1))
    return normaldataset,ranges,minv

def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix(r'KNN\datingTestSet2.txt')
    normMat,ranges,minv=autoNorm(datingDataMat)
    m=normMat.shape[0]
    # 取10%作测试数据
    numTestCount=int(m*hoRatio)
    errorcount=0
    for i in range(numTestCount):
        classify0Result=classify0(normMat[i,:],normMat[numTestCount:m,:],datingLabels[numTestCount:m],3)
        print(f'the classifier came back with:{classify0Result},the real answer is {datingLabels[i]}')
        if(classify0Result != datingLabels[i]):
            errorcount +=1
    print(f'the total error rate is: {errorcount/float(numTestCount)}')

def img2vertor(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        linestr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(linestr[j])
    return returnVect

def handwritingclasstest():
    hwLabels=[]
    # 返回目录下所有文件信息的列表
    trainingFileList=listdir(r'KNN\trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        filenamestr=trainingFileList[i]
        filestr=filenamestr.split('.')[0]
        classnumstr=int(filestr.split('_')[0])
        # 获取标签
        hwLabels.append(classnumstr)
        # 获取样本数据集
        trainingMat[i,:]=img2vertor(os.path.join(r'KNN\trainingDigits',filenamestr))
    testFileList=listdir(r'KNN\testDigits')
    errorcount=0
    mTest=len(testFileList)
    for i in range(mTest):
        file=testFileList[i]
        filename=file.split('.')[0]
        classnum=int(filename.split('_')[0])
        vectorundertest=img2vertor(os.path.join(r'KNN\testDigits',file))
        result=classify0(vectorundertest,trainingMat,hwLabels,3)
        print(f'the classifier came back with:{result},the real answer is {classnum}')
        if(result != classnum):
            errorcount+=1
    print(f'the total error rate is: {errorcount/float(mTest)}')




















