# # __len__
# class B:
#     def __len__(self):
#         print(666)
#         return 555

# obj=B()
# print(len(obj))

# __hash__
# class A:
#     pass

# o=A()
# print(hash(o))




# class A:
#     def __init__(self):
#         pass
#     # 类似java 重写tostring
#     def __str__(self):
#         print(666)
#         return 'xiaoer'
# a=A()
# print(a)=print(str(a))
# str(a)



# repr
# print('我叫%s'%('alex'))
# print('我叫%r'%('alex'))



# class A:
#     def __init__(self):
#         self.name='xiaoer'
#         pass
#     # str比str优先级高
#     def __repr__(self) :
#         print(111)
#         return f'name:{self.name}'
#     def __str__(self):
#         return '222'

# a=A()
# print(a)

# #__call__
# # 对象()自动触发对象从属于类（父类）的call方法
# class B:
#     def __init__(self):
#         pass
#     def __call__(self,*args,**kwargs):
#         print('__call__')
# obj=B()
# obj()

# # __eq__ 
# class A:
#     def __init__(self):
#         self.a=1
#         self.b=2
#     def __eq__(self,obj):
#         if self.a==obj.a and self.b == obj.b:
#             return True
# a=A()
# b=A()
# 自动调用对象中的eq方法
# print(a==b)

# # __del__析构方法
# class A:
#     def __init__(self):
#         self.a=1
#         self.b=2
#     def __eq__(self,obj):
#         if self.a==obj.a and self.b == obj.b:
#             return True
#     def __del__(self):
#         print(666)
# a=A()
# del a

# class A:
#     def __init__(self):
#         self.a=1
#         self.b=2
#     # 创建对象时，先执行new 在执行init
#     def __new__(cls,*args,**kwargs):
#         print('in new function')
#         return object.__new__(A,*args,**kwargs)
# a=A()

# 设计模式
# 单例模式：一个类只允许实例化一个对象
# class A:
#     pass

# o=A()
# print(o)
# o1=A()
# print(o1)


class A:
    __instance=None

    def __new__(cls,*args,**kwargs):
        if not cls.__instance:
            cls.__instance=object.__new__(cls)
            return cls.__instance
        return cls.__instance
obj=A()
print(obj)
obj1=A()
print(obj1)








