import cv2
import matplotlib.pyplot as plt
# print(cv2.__version__)
# 默认读取的格式是bgr
img=cv2.imread(r'opencv/1.jpg')
img2=cv2.imread(r'opencv/2.jpg')
# print(img)

def cv_show(img):
    cv2.imshow('a',img)
    # 按下任意键终止显示 或者显示n毫秒后消失
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 显示顺序为bgr
# print(img.shape)
# 读取灰度图
# img=cv2.imread(r'opencv/1.jpg',cv2.IMREAD_GRAYSCALE)
# cv_show('img',img)
# 保存
# cv2.imwrite(r'opencv\copy_1.jpg',img)
# 切割
# cat=img[0:50,0:200]
# cv_show('aa',img)
def change(img):
    b,g,r=cv2.split(img)
    img=cv2.merge((r,g,b))
    return img
# print(img.shape)
# i=img.copy()
# i[:,:,0]=0
# i[:,:,1]=0
# cv_show('r',i)
# b,g,r=cv2.split(img)
# img=cv2.merge((r,g,b))
# # 边界填充
# top_size,bottom_size,left_size,right_size=(50,50,50,50)
# replicate=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REPLICATE)
# reflect=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REFLECT)
# reflect101=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REFLECT101)
# warp=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_WRAP)
# constant=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_CONSTANT,value=0)

# import matplotlib.pyplot as plt
# plt.subplot(231)
# plt.imshow(img)
# plt.title('original')

# plt.subplot(232)
# plt.imshow(replicate,'gray')
# plt.title('replicate')

# plt.subplot(233)
# plt.imshow(reflect,'gray')
# plt.title('reflect')

# plt.subplot(234)
# plt.imshow(reflect101,'gray')
# plt.title('reflect101')

# plt.subplot(235)
# plt.imshow(constant,'gray')
# plt.title('constant')

# plt.subplot(236)
# plt.imshow(warp,'gray')
# plt.title('warp')
# plt.show()

# img2=img
# # 超过255取模
# i=img2+img
# # 超过255取255
# i=cv2.add(img,img2)
# print(i[:5,:,0])
# cv_show('a',i)

# print(img.shape)
img2=cv2.imread(r'opencv\2.jpg')
# print(img2.shape)
# img2=cv2.resize(img2,(200,200))
# cv_show(img2)
# img2=cv2.resize(img2,(10,100),fx=5,fy=1)
# cv_show(img2)
# 0.4,0.6表示两个图片的权重，0是偏置项，表示亮度
# res=cv2.addWeighted(img,0.2,img2,0.8,1)
# cv_show(res)

# src 原始图像
# thresh 阈值0-255（一般127）
# maxval 最大的可能值 255
# type 选择的方法
# ret,dest=cv2.threshold(src,thresh,maxval,type)
# 大于这个阈值取最大值（maxval）否则取9
cv2.THRESH_BINARY
# 和上面相反
cv2.THRESH_BINARY_INV 
# 大于阈值的部分设为阈值，否则不变
cv2.THRESH_TRUNC 
# 大于阈值的不变，小于阈值的为0
cv2.THRESH_TOZERO
# 大于阈值的为0，小于阈值的不变
cv2.THRESH_TOZERO_INV

img=change(img)
ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
title=['origin','binary','binary_inv','trunc','tozero','tozero_inv']
images=[img,thresh,thresh1,thresh2,thresh3,thresh4]
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],'gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

# # 读取视频
# vc=cv2.VideoCapture(r'opencv/video.mp4')
# if vc.isOpened():
#     open,frame=vc.read()
# else:
#     open=False
# while open:
#     ret,frame=vc.read()
#     if frame is None:
#         break
#     if ret ==True:
#         gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         cv2.imshow('result',gray)
#         if cv2.waitKey(100) and 0xFF==27:
#             break
# vc.release()
# cv2.destroyAllWindows()





