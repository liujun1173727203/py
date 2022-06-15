from cProfile import label
import ordertree
# dataset,labels=ordertree.createDataset()
# # ordertree.calcShannonEnt(dataset)
# # retdataset=ordertree.splitDataset(dataset,0,0)
# la=labels.copy()
# mytree=ordertree.createTree(dataset,la)
# # print(mytree)
# ordertree.createPlot(mytree)
# # ordertree.getnumleafs(mytree)
# ans=ordertree.classify(mytree,labels,[1,1])
# print(ans)


# 隐形眼镜数据集测试
fr=open(r'决策树\lenses.txt')
dataset=[example.strip().split('\t') for example in fr.readlines()]
labels=['age','prescript','astigmatic','tearRate']
lensestree=ordertree.createTree(dataset,labels)
ordertree.createPlot(lensestree)
