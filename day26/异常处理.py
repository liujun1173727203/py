import errno


# try:
#     num=int(input('>>>'))

# # except只执行一个
# except ValueError:
#     print('输入元素非数字')
# except KeyError:
#     print('没有此建')
# finally:
#     print('111')


class liujunerror(BaseException):
    def __init__(self,msg):
        self.msg=msg
    def __str__(self):
        return self.msg

try:
    raise liujunerror('leixingerror')
except liujunerror as e:
    print(e)