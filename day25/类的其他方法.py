# 类方法
from django.urls import clear_script_prefix


# class A:
#     def func(self):
#         print("实例方法")
#     @classmethod
#     def cls_func(cls):
#         print(f'cls--->{cls}')
#         print('类方法')

# A.cls_func()
# obj=A()
# obj.cls_func()
# 类方法：通过类名调用，自动将类名地址传给cls
# 可以通过调用也可以，但传入的地址还是类名地址
# 类方法的作用：
#     1.得到类名可以实例化对象
#     2.可以操作类的属性


# 记录创建对象个数
# class Stu:
#     count=0
#     def __init__(self):
#         Stu.addcount()
#         pass
#     @classmethod
#     def addcount(cls):
#         cls.count=cls.count+1
    
#     @classmethod
#     def getcount(cls):
#         return cls.count

# a=Stu()
# print(Stu.getcount())

# # 静态方法
# class Stu:
#     count =0
#     def __init__(self):
#         Stu.addcount()
#         pass
#     def func(self):
#         print("实例方法")
#     @classmethod
#     def addcount(cls):
#         cls.count=cls.count+1
#         print("类方法")
#     @staticmethod
#     def func1():
#         print("静态方法——不依赖于类")
# # 将动态方法伪装成属性
#     @property
#     def weight(self):
#         print("weight")

# obj =Stu()
# # print(obj.weight())
# print(obj.weight)

# property是一个组合

class Foo:
    @property
    def weight(self):
        print("weight")

    @weight.setter
    def  weight(self,value):
        print(f'setter{value}')
    
    @weight.deleter
    def weight(self):
        print("delete")

obj=Foo()
obj.weight
obj.weight=11
del obj.weight