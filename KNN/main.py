from cProfile import label
from tabnanny import verbose
from tokenize import group

from numpy import array, tile
import KNN
# group,labels=KNN.createDataSet()
# # print(group,labels)
# # q=group.shape[0]
# # # print(group.shape)
# # a=tile((1,1),(q,1))-group
# # print(a)
# # print(a.sum(axis=1))
# print(KNN.classify0((0.5,0.1),group,labels,4))

dataset,vetor=KNN.file2matrix(r'KNN\datingTestSet2.txt')
# print(dataset,vetor)
dataset,rangw,minv=KNN.autoNorm(dataset)
import matplotlib
import matplotlib.pyplot as plt
fig =plt.figure()
# add_subpot添加画布
ax=fig.add_subplot(111)
# dataset[a,b]几行几列 :表示全部行，1表示第一列
ax.scatter(dataset[:,0],dataset[:,1],15.0*array(vetor),array(vetor))
# print(15.0*array(vetor))
plt.show()


# KNN.datingClassTest()
# KNN.handwritingclasstest()



