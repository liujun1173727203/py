from ast import operator
from cProfile import label
from enum import unique
from functools import reduce
from math import log
from venv import create

def createDataset():
    dataset=[
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,0,'yes'],
        # [1,1,'maybe'],
    ]
    labels =['no surfacing','flippers']
    return dataset,labels

# 熵的计算:l=sum(-p*log2(p)) 为了查看信息增益
def calcShannonEnt(dataset):
    numEntries =len(dataset)
    labelscount = {}
    for featVec in dataset:
        currentLabel =featVec[-1]
        if currentLabel not in labelscount.keys():
            labelscount[currentLabel]=0
        labelscount[currentLabel]+=1
    shannonEnt=0
    for key in labelscount:
        prob =float(labelscount[key]/numEntries)
        shannonEnt -=prob*log(prob,2)
    return shannonEnt

# 待划分数据集,划分数据集的特征,返回的值
def splitDataset(dataset,axis,value):
    retdataset=[]
    for featVec in dataset:
        if featVec[axis]==value:
            reduceFeatvec=featVec[:axis]
            reduceFeatvec.extend(featVec[axis+1:])
            retdataset.append(reduceFeatvec)
    return retdataset

def chooseBestFeatureToSplit(dataset):
    numfeatures =len(dataset[0])-1
    baseentropy=calcShannonEnt(dataset)
    bestInfogain=0
    bestfeature=-1
    for i in range(numfeatures):
        featlist=[example[i] for example in dataset]
        uniquevals=set(featlist)
        newentropy=0
        for value in uniquevals:
            subdataset=splitDataset(dataset,i,value)
            #?
            prob =len(subdataset)/float(len(dataset))
            newentropy+=prob*calcShannonEnt(subdataset)
        infogain =baseentropy-newentropy
        if(infogain>bestInfogain):
            bestInfogain=infogain
            bestfeature =i
    return bestfeature 


def majoritycnt(classlist):
    classcount={}
    for vote in  classlist:
        if vote not in classcount.keys():
            classcount[vote]=0
        classcount+=1
    sortedclasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

def createTree(dataset,labels):
    # 取每个数据项的最后一项,即类标签加入到list中
    classlist=[example[-1] for example in dataset]
    if classlist.count(classlist[0])==len(classlist):
        # 相等说明类别相同停止划分
        return classlist[0]
    # 递归出口
    if len(dataset[0])==1:
        return majoritycnt(classlist)
    bestfeat=chooseBestFeatureToSplit(dataset)
    bestfeatlabel=labels[bestfeat]
    mytree ={bestfeatlabel:{}}
    del(labels[bestfeat])
    featvalues=[example[bestfeat] for example in dataset]
    uniquevalues=set(featvalues)
    for value in uniquevalues:
        sublabels =labels[:]
        mytree[bestfeatlabel][value]=createTree(splitDataset(dataset,bestfeat,value),sublabels)
    return mytree

import matplotlib.pyplot as plt

# 决策节点
decisionnode=dict(boxstyle='sawtooth',fc='0.8')
# 叶节点
leafnode=dict(boxstyle='round4',fc='0.8')
arrow_args=dict(arrowstyle='<-')
def plotmidtext(cntrpt,parentpt,txtstring):
    xmid=(parentpt[0]-cntrpt[0])/2+cntrpt[0]
    ymid=(parentpt[1]-cntrpt[1])/2+cntrpt[1]
    print(f'axis({xmid},{ymid})_{txtstring}')
    createPlot.ax1.text(xmid,ymid,txtstring)

def plotTree(mytree,parentpt,nodetxt):
    numleaf=getnumleafs(mytree)
    depth=gettreedepth(mytree)
    firststr=list(mytree.keys())[0]
    cntrpt=(plotTree.xoff+(1.0+float(numleaf))/2/plotTree.totalW,plotTree.yoff)
    plotmidtext(cntrpt,parentpt,nodetxt)
    plotnode(firststr,cntrpt,parentpt,decisionnode)
    seconddict=mytree[firststr]
    plotTree.yoff=plotTree.yoff-1/plotTree.totalD
    for key in seconddict.keys():
        if type(seconddict[key]).__name__=='dict':
            plotTree(seconddict[key],cntrpt,str(key))
        else:
            plotTree.xoff=plotTree.xoff+1/plotTree.totalW
            plotnode(seconddict[key],(plotTree.xoff,plotTree.yoff),cntrpt,leafnode)
            plotmidtext((plotTree.xoff,plotTree.yoff),cntrpt,str(key))
    plotTree.yoff=plotTree.yoff+1/plotTree.totalD

def plotnode(nodetxt,centerpt,parentpt,nodetype):
    # print(f'{parentpt},{centerpt},{nodetxt},{nodetype}')
    createPlot.ax1.annotate(nodetxt,xy=parentpt,
    xycoords='axes fraction',
    xytext=centerpt,textcoords='axes fraction',
    va='center',ha='center',
    bbox=nodetype,arrowprops=arrow_args)
def createPlot(intree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    # subplot()将画布分为一行一列，当前位置为1
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getnumleafs(intree))
    plotTree.totalD=float(gettreedepth(intree))
    plotTree.xoff=-0.5/plotTree.totalW
    plotTree.yoff=1
    plotTree(intree,(0.5,1),'')
    plt.show()

def getnumleafs(mytree):
    numleafs=0
    firststr=list(mytree.keys())[0]
    seconddict=mytree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__=='dict':
            numleafs +=getnumleafs(seconddict[key])
        else: numleafs +=1
    return numleafs

def gettreedepth(mytree):
    maxdepth=0
    firststr=list(mytree.keys())[0]
    seconddict=mytree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__=='dict':
            thisdepth=gettreedepth(seconddict[key])+1
        else: thisdepth =1
        if thisdepth>maxdepth:
            maxdepth=thisdepth
    return maxdepth 


def classify(inputtree,featlabels,testvec):
    # print(featlabels)
    firststr=list(inputtree.keys())[0]
    seconddict=inputtree[firststr]
    featindex=featlabels.index(firststr)
    for key in list(seconddict.keys()):
        if testvec[featindex]==key:
            if type(seconddict[key]).__name__=='dict':
                classlabel=classify(seconddict[key],featlabels,testvec)
            else:
                classlabel=seconddict[key]
                # print()
    return classlabel
