import logging
import logging.handlers


# def get_logger(name='root'):
#     logging.basicConfig(filename=name, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                         level=logging.INFO)
#     handler = logging.StreamHandler()
#     filer = logging.handlers.RotatingFileHandler(name, maxBytes=10485760, backupCount=20, encoding="utf-8")
#     # handler.setFormatter(formatter)
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
#     logger.addHandler(handler)
#     logger.addHandler(filer)
#     return logger

import logging
import os
import time
from logging.handlers import RotatingFileHandler

import os
import time
import logging
from logging.handlers import RotatingFileHandler
from glob import glob
class CustomRotatingFileHandler(RotatingFileHandler):
    def __init__(self, base_filename, maxBytes, backupCount, encoding=None, delay=False):
        # 确保 maxBytes 是一个整数
        maxBytes = int(maxBytes)
        super().__init__(base_filename, maxBytes=100*1024*1024, backupCount=7,encoding=None, delay=False)
        self.base_filename = base_filename
        self.backupCount = backupCount
        self.base, self.ext = os.path.splitext(self.baseFilename)

    def doRollover(self):
        # 调用父类的 doRollover 方法以执行标准的轮转逻辑
        super().doRollover()

        # 获取当前时间戳用于日志文件名
        current_time = time.strftime("%Y%m%d%H%M%S")

        # 计算新日志文件的名称
        base, ext = os.path.splitext(self.baseFilename)
        new_base = f"{base}.{current_time}"
        new_filename = new_base if not ext else f"{new_base}{ext}"

        # 重命名当前日志文件为带有时间戳的新文件
        if os.path.exists(self.baseFilename):
            try:
                os.rename(self.baseFilename, new_filename)
            except OSError as e:
                print(f"Error renaming file: {e}")
                # 可以选择在这里处理错误，比如记录到另一个日志等

        # 更新 baseFilename 以指向新的（或即将写入的）日志文件
        # 注意：这里我们不改变 self.stream，因为 super().doRollover() 已经处理了
        self.baseFilename = new_filename if self.backupCount == 0 else self.base_filename
        # 如果日志文件数量超过 backupCount，则删除最旧的
        log_files = glob(f"{self.base}*.*")
        if len(log_files) > self.backupCount + 1:  # +1 是因为当前正在写入的文件也算一个
            # 对文件名进行排序，找到最旧的日志文件（这里假设文件名中的时间戳是唯一的且按时间顺序排列）
            log_files.sort()
            oldest_file = log_files[0]
            try:
                os.remove(oldest_file)
                print(f"Deleted oldest log file: {oldest_file}")
            except OSError as e:
                print(f"Error deleting file: {e}")

                # 配置日志记录（注意：这里应该放在使用 handler 的代码之前）
def get_logger(log_path,task_id=None):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # 创建自定义处理器（注意：maxBytes 应该是一个整数）
    handler = CustomRotatingFileHandler(
        log_path,100*1024*1024,7
    )
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    if task_id:
        def log_with_task_id(level, msg):
            full_msg = f"{task_id}- {msg} "
            getattr(logger, level)(full_msg)  # 使用getattr来调用正确的日志级别方法
        for level in ['debug', 'info', 'warning', 'error', 'critical']:
            setattr(logger, f'info', lambda msg: log_with_task_id(level, msg))

    # 将处理器添加到记录器中
    return logger

