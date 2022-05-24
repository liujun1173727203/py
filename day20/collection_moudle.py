from collections import deque, namedtuple
import imp
Point =namedtuple('Point',['x','y'])
p=Point(1,1)
print(p)
print(p.x)
print(p[0])

# deque 双端队列
from collections import deque
q=deque([1,2,3,4,5])
q.append('a')
print(q)
q.appendleft('b')
print(q)
q.pop()
print(q)
q.popleft()
print(q)
# 按照索引插入
q.insert(1,'s')
print(q)

from collections import defaultdict
li=[11,2,3,4,5,56,67,6]
# defaultdict()参数必须可迭代
dic =defaultdict(list)
for i in li:
    if i<25:
        dic['k1'].append(i)
    else:
        dic['k2'].append(i)

print(dic)


from collections import Counter
# 每个元素出现的次数
c=Counter('asdasdada')
print(c)