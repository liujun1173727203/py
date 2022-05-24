import logging

# 限制  不能文件和屏幕输出不能同时输出
# 修改日志等级
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    filename=r'logging.log',
    # format='%(asctime)s %(filename)s %(levelname)s %(message)s'
)

logging.debug("debug")
logging.info("info")
logging.warning("warn")
logging.error("error")
logging.critical("critical")




