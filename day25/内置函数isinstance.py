# class A:
#     pass
# class B(A):
#     pass
# obj=B()
# 判断对象与类的关系
# isinstance(a,b) 判断a是否是b类，或者派生类
# print(isinstance(obj,B))
# print(isinstance(obj,A))

class A:
    pass
class B(A):
    pass

class C(B):
    pass
# issubclass(a,b)判断a类是否是b类的子孙类 或者派生类
print(issubclass(B,A))
