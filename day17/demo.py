# import json
# lis =[i for i in range(100)]
# with open('jsontest.json','w',encoding='utf-8') as f:
#     f.write(json.dumps(lis))
# with open('jsontest.json','r',encoding='utf-8') as f:
#     a=f.readline()
#     print(type(a))
#     b=json.loads(a)
#     print(type(b)

# json pickle 一个转化为字符串  一个转化为byte类型
# dump load 文件操作
# dumps loads 网络传输



import pickle
lis =[i for i in range(100)]
with open('pickledemo.pickle','wb') as f:
    pickle.dump(lis,f)

with open('pickledemo.pickle','rb') as f:
    a=pickle.load(f)
    print(a,type(a))