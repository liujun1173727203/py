'''
Description: 
Author: xiaoer
Date: 2022-06-15 15:48:51
LastEditTime: 2022-07-01 12:01:29
LastEditors:  
'''
import random
from turtle import position
from xml.dom.minidom import Document

from numpy import array, log, ones, zeros


def loaddata():
    postingList =[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createvocablist(dataset):
    vocabset =set([])
    for document in dataset:
        vocabset =vocabset | set(document)
    return list(vocabset)
def setodwords2vec(vocablist,inputset):
    returnvec =[0]*len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] +=1
            returnvec[vocablist.index(word)] =1
        else: print(f'the word:{word} is not in my vocabulary')
    return returnvec

def trainNB(trainmatrix,traincategory):
    numTrainDocs=len(trainmatrix)
    numWords=len(trainmatrix[0])
    pAbusvie=sum(traincategory)/float(numTrainDocs)
    p0num=ones(numWords)
    p1num=ones(numWords)
    p0denom=2
    p1denom=2
    for i in range(numTrainDocs):
        if traincategory[i]==1:
            p1num+=trainmatrix[i]
            p1denom+=sum(trainmatrix[i])
        else:
            p0num+=trainmatrix[i]
            p0denom+=sum(trainmatrix[i])
    # 每一个词向量在该类别的概率
    p1vect=log(p1num/p1denom)
    p0vect=log(p0num/p0denom)
    return p0vect,p1vect,pAbusvie

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # print(vec2Classify)
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    #概率相乘再相加
    p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)
    if p1>p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts,listClasses = loaddata()
    myVocabList = createvocablist(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setodwords2vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation','stupid']
    thisDoc = array(setodwords2vec(myVocabList,testEntry))
    print(f'{testEntry},classified as: {classifyNB(thisDoc,p0V,p1V,pAb)}')
    testEntry = ['stupid','garbage']
    thisDoc = array(setodwords2vec(myVocabList,testEntry))
    print(f'{testEntry},classified as: {classifyNB(thisDoc, p0V, p1V, pAb)}')


def textParse(bigstring):
    import re
    listofTokens=re.split(r'\W',bigstring)
    return [t.lower() for t in listofTokens if len(t)>0]
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open(f'beiyes\email\spam\{i}.txt').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open(f'beiyes\email\spam\{i}.txt').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)

    vocabList = createvocablist(docList)
    #建立词列表
    print(range(50))
    print(list(range(50)))
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        #随机一个数字，并将对应文章放入测试集
        del(trainingSet[randIndex])
        #删除该数字
    #随机测试集
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setodwords2vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pAb = trainNB(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setodwords2vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pAb) != classList[docIndex]:
            errorCount+=1
    print(f'the error rate is :{float(errorCount)/len(testSet)}')


