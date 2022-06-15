from pyclbr import Function
from types import FunctionType, MethodType


class A:
    def func(self):
        pass
def func1():
    pass
# 通过类名调用的方法叫函数
print(func1)
print(A.func)
obj=A()
# 通过对象调用的叫方法
print(obj.func) 
print(isinstance(func1,FunctionType))
print(isinstance(obj.func,MethodType))








# 函数都是显性传参
# 方法都是隐形传参