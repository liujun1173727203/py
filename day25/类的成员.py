class demo():
    # 公有
    name='xiaoer'
    # 私有
    __name='zjds'
    def __init__(self):
        pass
    def func1(self):    #普通方法
        print(self.name)
        print(self.__name)
    def __func2():          #只能在内部方法调用
        pass

class B(demo):
    def func(self):
        print(self.name)
        # print(self.__name)            #不可访问


obj=demo()
obj.func1()
#不行
# print(obj.__name)
类在加载时，只要遇到类中的私有成员，都会在私有成员前面加上_类名
print(demo._demo__name)
obj1=B()
obj1.func()