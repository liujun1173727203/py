import random
# 0-1之间
print(random.random())
# 生成之间的随机数
random.uniform(1,6)
# [1,5]
print(random.randint(1,5))

# [1,10) 3为步长
print(random.randrange(1,10,3))

# 可迭代对象
# 在指定列表随机选择
print(random.choice(['12','23d','21']))
# 控制选几个
print(random.sample(['a','s','d'],2))

# 打乱顺序
item=[i for i in range(10)]
random.shuffle(item)
print(item)





