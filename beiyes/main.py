'''
Description: 
Author: xiaoer
Date: 2022-06-18 10:48:46
LastEditTime: 2022-06-21 11:51:26
LastEditors:  
'''
import re

from numpy import array
import beiyes

# listoposts,listclasses =beiyes.loaddata()
# myvocablist =beiyes.createvocablist(listoposts)
# # vec=beiyes.setodwords2vec(myvocablist,listoposts[0])
# trainmat=[]
# for postindoc in listoposts:
#     trainmat.append(beiyes.setodwords2vec(myvocablist,postindoc))
# p0v,p1v,pAb=beiyes.trainNB(trainmat,listclasses)
# # print(p0v,p1v,pAb)
# # beiyes.testingNB()
# txt =open(r'朴素贝叶斯算法\email\ham\1.txt').read()
# # txt=''
# # for line in file.readlines():
#     # txt +=line.strip(r'\n')
# reggx=re.compile( r"\W")
# txt=reggx.split(txt.lower())
# txt=[data for data in txt if len(data)>0]
# aimdata=array(beiyes.setodwords2vec(myvocablist,txt))
# res=beiyes.classifyNB(aimdata,p0v,p1v,pAb)
# print(res)
# # print(txt)
beiyes.spamTest()