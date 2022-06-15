# 正则表达式
from re import A


# \A ^从头开始匹配
# \Z &从尾开始匹配
import re
print(re.findall('\n','sadsada\nasfsfs\nfs'))

# 元字符
#  . ? * + {m,n} .* .*?
#  . 匹配任意一个字符   如果匹配成功 则光标移动到匹配后的最后
print(re.findall('a.b','asdasbasaafsa'))

# ? 匹配0个或1个左边字符定义的片段 
print(re.findall('a?b','ab aab'))

# * 匹配0个或多个左边字符 满足贪婪
print(re.findall('a*b','ab b aaaaab'))

# + 匹配1个或多个
print(re.findall('a+b','ab b aaaaab'))

# {m,n}匹配m个到n个左边字符[m,n]两边都能娶到
print(re.findall('a{1,5}b','abavavaa'))

# . 任意一个字符
# .*任意字符的任意数量个字符
print(re.findall('a.*b','ab aabbbbbab'))


# .*? 限定遵从非贪婪匹配
# 注意区别
print(re.findall('a.*?b','ab acbbbbb aaab'))
print(re.findall('a.*b','ab acbbbbb aaab'))

# []匹配括号中的任意一个字符
print(re.findall('a[abc]b','ab aab abb acb'))
# - 表示范围
print(re.findall('a[a-c]b','ab aab abb acb'))
print(re.findall('a[a-zA-Z]b','ab aab abb acb'))
# 当想匹配-时，不能放中间
print(re.findall('a[-a]b','a-b aab abb acb'))
# ^在中括号中表示取反 必须放在最前面
print(re.findall('a[^0-9]','a-b a1b aab'))
# \w 一个字母数字或下划线
# 带括号只返回括号中的
print(re.findall('(\w+)_sb','asda_sb asd312a_sb'))
print(re.findall('\w+_sb','asda_sb asd312a_sb'))

# | 竖线两边的任意一部分
print(re.findall('as|ddd|sada','dddsada asddd asdddsada'))
# 返回整体
print(re.findall('compan(?:y|ies)','sadasdasdad'))

# 查找第一个  
print(re.search('sb|alex','alex sb sdada'))
# 通过group返回
print(re.search('sb|alex','alex sb sdada').group())

# match 从字符串开头匹配 符合返回开头，否则None
print(re.match('lex','alex sb sdada'))


# 指定不同分割符进行分割
print(re.split('[;,./]',''))

print(re.sub('旧的','新的','旧的字符串'))

obj =re.compile('as')  
print(obj.findall('asdasdada'))

# 返回可迭代对象
ret =re.finditer('\d','2312312312131')









