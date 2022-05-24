print("__init__")
from demo import demo1
# 绝对导入
from demo.demo1 import f1,f2
from demo.demo2 import f3,f4
from demo.demo3 import f5,f6
# 相对导入  .为当前目录
from .demo1 import f1,f2
from .demo2 import f3,f4
from .demo3 import f5,f6
from .aaa.ass import f7 