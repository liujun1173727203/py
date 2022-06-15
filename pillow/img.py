from PIL import Image,ImageFilter,ImageChops,ImageEnhance
from numpy import imag, size
# im =Image.open(r'pillow\1.jpg')
# # im.show()
# print(im.format)
# print(im.size)
# print(im.height)
# # 获取像素点颜色
# print(im.getpixel((100,100)))


# 透明度
# im =Image.open(r'pillow\1.jpg').convert(mode='RGB')
# # 创建一块红布
# img2=Image.new('RGB',im.size,'red')
# # aplha决定im，img2的比例
# Image.blend(im,img2,alpha=0.5).show()


# # 遮罩混合

# i1=Image.open(r'pillow\1.jpg')
# i2=Image.open(r'pillow\2.jpg')
# i2.resize(i1.size)
# r,g,b=i2.split()
# Image.composite(i1,i2,r).show()

# 按像素缩放图片
# i1=Image.open(r'pillow\1.jpg')
# Image.eval(i1,lambda x:x*3).show()

# # 按尺寸缩放
# i1=Image.open(r'pillow\1.jpg')
# l2=i1.copy()
# l2.thumbnail((200,200))
# l2.show()

# i1=Image.open(r'pillow\1.jpg')
# i2=i1.copy()
# # 剪切
# region=i1.crop((0,0,120,120))
# # 粘贴
# i2.paste(region,(200,100))
# i2.show()


# 旋转
i1=Image.open(r'pillow\1.jpg')
i1.rotate(45).show()


# i1=Image.open(r'pillow\1.jpg')
# # 滤镜上下左右旋转
# i1.transpose(Image.FLIP_LEFT_RIGHT).show()


i1=Image.open(r'pillow\1.jpg')
i2=Image.open(r'pillow\2.jpg')
i2=i2.resize(i1.size)
r1,b1,g1=i1.split()
r2,b2,g2=i2.split()
# 通过像素合并
# im=Image.merge('RGB',[r2,g1,b1]).show()


# 滤镜 进行模糊
# i1.filter(ImageFilter.GaussianBlur).show()

# 图片加法运算
# ImageChops.add(i1,i2).show()

# 减法
# ImageChops.subtract(i1,i2).show()
# 取暗部
# ImageChops.darker(i1,i2).show()
# 亮部
# ImageChops.lighter(i1,i2).show()
# 两张图片相互叠加
# ImageChops.multiply(i1,i2).show()

# 投影
# ImageChops.screen(i1,i2).show()
# 反色
# ImageChops.invert(i1).show()



# w,h=i1.size
# im_out=Image.new('RGB',(9*w,h))

# im_out.paste(i1,(0,0))
# # # 色彩增强
# im_color=ImageEnhance.Color(i1)
# imgb=im_color.enhance(1.5)
# im_out.paste(imgb,(w,0))
# imgb=im_color.enhance(0.5)
# im_out.paste(imgb,(w*2,0))

# # 亮度增强
# nbb=ImageEnhance.Brightness(i1)
# l=nbb.enhance(1.5)
# im_out.paste(l,(3*w,0))
# im_out.show()

# 通过像素点调整亮度
# w,h=i1.size
# im_out=Image.new('RGB',(9*w,h))
# b=i1.point(lambda x:x*2)
# im_out.paste(b,(w,0))
# b=i1.point(lambda x:x*0.2)
# im_out.paste(b,(w*2,0))
# im_out.show()

# from PIL import ImageDraw,ImageFont
# a =Image.new('RGB',(200,200),'white')
# d =ImageDraw.Draw(a)
# d.rectangle((50,50,150,150),outline='red')
# d.text((60,60),'first',fill='green')
# a.show()

# ImageFont.load(filename)
# # 不指定当前路径，从sys中找
# ImageFont.load_path(filename)
# # 指定文件
# ImageFont.truetype(file,size)


# i=Image.open(r'pillow\1.jpg')
# w,h=i.size
# D=ImageDraw.ImageDraw(i)
# D.arc((100,100,w,h),0,360,fill='blue')
# i.show()
# # 绘制十字
# d=ImageDraw.Draw(i)
# d.line((0,0,w,h),fill=(255,0,0),width=3)
# d.line((0,h,w,0),fill=(255,0,0),width=3)
# i.show()

# # 按色素调节
# def getColor(oc):
#     if(oc[0]>60 and oc[1]>60):
#         return (oc[0],oc[1],0)
#     return oc
# d=ImageDraw.Draw(i)
# for x in range(w):
#     for y in range(h):
#         oc=i.getpixel((x,y))
#         d.point((x,y),fill=getColor(oc))
# i.show()





