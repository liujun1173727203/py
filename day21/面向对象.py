from abc import ABCMeta, abstractmethod
from pickle import PERSID
from re import S


# class Human:
#     a='bbb'
#     b='bbb'

#     def __init__(self,d):
#         self.c='ccc'
#         self.d=d
#         print(self)
#         print(123)

#     def aa(self,e):
#         self.e=e
#         print(f"{self.e}aa")
#     def bb(self):
#         print("bb")
# # 实例化对象
# obj =Human('ddd')

# print(obj.__dict__)

# obj.aa('eee')

# class nan:
#     def __init__(self,name):
#         self.name=name
#     def meet(self,girl=None):
#         self.girl=girl
#     def dinner(self):
#         if self.girl:
#             print(f'{self.name} with {self.girl}')
#         self.girl.shop(self)
        
# class girl:
#     def __init__(self,name):
#         self.name=name
#     def shop(self,nan_self):
#         print(f'{nan_self.name}{self.name}shopping')
# n =nan('x')
# v=girl('v')
# n.meet(v)
# n.dinner()

# class animal:
#     def __init__(self,name):
#         self.name=name
#     def food(self):
#         print("food")
# # 继承
# class person(animal):
#     eat='e'

#     def __init__(self,age,name):
#         # 第一种
#         # animal.__init__(self,name)
#         # self.age=age
#         # 第二种
#         super().__init__(name)
#         self.age=age
#     def eat(self):
#         print("eat")
#         super().food()


# # 子类只能调用父类  不能修改

# # 对象查找顺序 对象空间 子类查找  父类查找


# p=person('xiaoer','a')
# p.eat()
# print(p.__dict__)




# # 多继承

# # class a:
# #     def __init__(self,a):
# #         self.a=a
# #     def ac(self):
# #         print("it is a")
# # class b:
# #     def __init__(self,b):
# #         self.b=b
# #     def ac(self):
# #         print("it is b")
# # class c(a,b):
# #     def __init__(self,c):
# #         self.c=c
# #     def ac(self):
# #         print("it is c")
# # demo=c("ccc")
# # 执行的c的ac（）
# # demo.ac()


# class a:
#     def __init__(self,a):
#         self.a=a
#     def ac(self):
#         print("it is a")
# class b:
#     def __init__(self,b):
#         self.b=b
#     def ac(self):
#         print("it is b")
# class c(a,b):
#     def __init__(self,c):
#         self.c=c
#     # def ac(self):
#     #     print("it is c")
# demo=c("ccc")
# # 执行a的ac()
# demo.ac()


# # 鸭子类型
# class A:
#     def login(self):
#         pass
#     def register(self):
#         pass
# class B:
#     def login(self):
#         pass
#     def register(self):
#         pass
# A B两个类没有关系，功能相似 py会讲相似的功能让其命名相同

# 举例
# Str.index()
# list.index()


# class A:
#     def f1(self):
#         print('f1')
#     def f2(self):
#         print('f2')
# class F00(A):
#     def f1(self):
#         super().f2()
#         print('foo')

# obj=F00()
# obj.f1()


# super()
class A:
    def f1(self):
        print('AA')

class B(A):
    def f1(self):
        super().f1()
        print('bb') 

class F00(A):
    def f1(self):
        # super().f1()
        print('foo')       
class D(B,F00):
    def f1(self):
        super().f1()
        print('DD')
obj=D()
obj.f1()
# D,B,f00,A
print(D.mro()) 
# super()严格意义上不是执行父类的方法
# 单继承：super() 肯定执行的父类的方法
# 多继承：super(S,self)严格按照self从属于的类的mor算法的执行顺序执行，S类的下一位



# 类的约束
# 1.在父类建立约束
# 2.利用抽象类的概念
# 第一种
# class payment():
#     def pay(self):
#         raise Exception("主动报错")     #约定接口名

# class  alipay(payment):
#     def pay(self,money):
#         print(f"ali{money}")

# class  zfbpay(payment):
#     def pay(self,money):
#         print(f"zfb{money}")

# def pay(obj,moeny):
#     obj.pay(moeny)
# obj3=alipay()
# obj2=zfbpay()
# pay(obj3,100)

class payment(metaclass=ABCMeta):
    # 抽象类        继承后没有此方法强制报错
    @abstractmethod
    def pay(self):
        raise Exception("主动报错")     #约定接口名

class  alipay(payment):
    def pay(self,money):
        print(f"ali{money}")

class  zfbpay(payment):
    def pay(self,money):
        print(f"zfb{money}")

def pay(obj,moeny):
    obj.pay(moeny)
obj3=alipay()
obj2=zfbpay()
pay(obj3,100)













