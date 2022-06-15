# 反射：指通过字符串去操作对象的方式

from itertools import count


# class A:
#     country='中国'
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age
#     def func(self):
#         print('in a func')
# obj=A('xr',12)
# print(hasattr(obj,'name'))
# print(hasattr(obj,'age'))
# print(hasattr(obj,'country'))

# print(getattr(obj,'name'))
# print(getattr(obj,'func'))
# f=getattr(obj,'func')
# f()

# print(getattr(obj,'sex',None))
# setattr(obj,'sex','nan')
# print(obj.__dict__)

# 操作其他模块
# import demo
# print(getattr(demo,'name'))


# 从本模块操作
# a='666'
# def func1():
#     print('int this moudle')

# import sys
# print(getattr(sys.modules[__name__],'a'))
# getattr(sys.modules[__name__],'func1')()

def fun1():
    print('func1')

def fun2():
    print('func2')

def fun3():
    print('func3')

def fun4():
    print('func4')

func_list=[f'fun{i}' for i in range(1,5)]
import sys
for func in func_list:
    getattr(sys.modules[__name__],func)()

