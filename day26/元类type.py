from tkinter.tix import DirTree
from regex import P


class A:
    pass

# 打印他从属的类
# type获取对象的从属于的类
# 大部分类都是由type元类实例化得来的
print(type(A))
print(type(dict))
print(type("asf"))

# type与object的关系
# object类是type类的一个实例
# object类是type类的父类
print(type(object))
print(issubclass(type,object))