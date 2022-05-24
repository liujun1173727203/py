import logging

logger =logging.getLogger()
# 创建文件对象
fh =logging.FileHandler('standard.log',encoding='utf-8')
# 创建屏幕对象
sh =logging.StreamHandler()
# 创建输出格式
formateer =logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
sh.setFormatter(formateer)
fh.setFormatter(formateer)

# 设计级别
# 先设置总的
logger.setLevel(10)

fh.setLevel(10)
sh.setLevel(40)


logger.addHandler(sh)
logger.addHandler(fh)

logging.debug("debug")
logging.info("info")
logging.warning("warn")
logging.error("error")
logging.critical("critical")