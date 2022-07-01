'''
Description: 
Author: xiaoer
Date: 2022-06-27 12:44:25
LastEditTime: 2022-06-30 17:38:36
LastEditors:  
'''
import logistic
dataarr,labelMat=logistic.loadDataSet()
# weights=logistic.gradAscent(dataarr,labelMat)
# weights=logistic.stocGradAscent0(dataarr,labelMat)
weights=logistic.stocGradAscent1(dataarr,labelMat)
logistic.plotBestFit(weights)