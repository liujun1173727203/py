import hashlib
import os

from pymysql import NULL
# #md5
# ret=hashlib.md5()
# ret.update('123'.encode('utf-8'))
# a=ret.hexdigest()
# print(a)

# # 不是绝对安全 通过撞库测试可以测试到

# # 加固定盐 增强安全性
# ret=hashlib.md5('xiaoer'.encode('utf-8'))
# ret.update('123'.encode('utf-8'))
# a=ret.hexdigest()
# print(a)


# # 加动态盐
# name=input()
# ret=hashlib.md5(name.encode('utf-8'))
# ret.update('123'.encode('utf-8'))
# a=ret.hexdigest()
# print(a)


# #sha系列  安全系数高， 耗时高
# # 也可以加盐
# ret=hashlib.sha512()
# ret.update('123'.encode('utf-8'))
# a=ret.hexdigest()
# print(a)


# # 文件一致性校验
# # 前后两次的加密信息一致则文件一致
# ret =hashlib.md5()
# # print(os.path.dirname(__file__))
# with open('test',mode='rb') as f:
#     content=f.read()
#     ret.update(content)
# print(ret.hexdigest())


# 当文件过大,进行分步加密
ret =hashlib.md5()
with open('test',mode='rb') as f:
    while 1:
        bts=f.read(128)
        print(bts)
        if not bts:
            break
        ret.update(bts)
print(ret.hexdigest())







