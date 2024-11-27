import base64
import os
import cv2
from algrithom.tool.logic import WarnLogic
import numpy as np
class WarnConfig:
    def __init__(self,param):
        self.alarm_last=param["alarm_last"]
        self.alarm_interval= param["alarm_interval"]
        self.alarm_classes=param["alarm_classes"]
        self.abnormalPercent=0.6
class AlgThread:
    def __init__(self):
        pass
    def logicInit(self):
        logic = WarnLogic(self.warnConfig.alarm_last,self.warnConfig.alarm_interval,
                          self.warnConfig.abnormalPercent);
        return logic
    def algorithmLogic(self,detect,timestamp):
        Logic=self.Logic
        warnFlag=0
        warnFlag = Logic.update(detect, timestamp)
        return warnFlag
    def sendkafkamsg(self,msg):
        if self.msgapp.kafka_send:
            msg["image"]=base64.b64encode(msg["image"]).decode('utf-8')
            msg = convert_ndarray_to_list(msg)
            topic_name =self.param['topic']
            # # 发送报警事件到Kafka
            self.logger.info("开始发送消息")

            future = self.msgapp.producer.send(topic_name, msg)
            # 尝试获取发送结果，同时处理可能的异常
            try:
                # 等待消息发送完成（或直到超时），这里假设超时时间为60秒
                result = future.get(timeout=10)
                self.logger.info(f'Message sent to {result.topic} [{result.partition}] offset {result.offset}')
                self.logger.info("发送kafka消息成功")
            except Exception as e:
                # 处理发送过程中出现的异常
                self.logger.info(f'Failed to send message: {e}')
        else:
            self.logger.info('kafka server can not use')


    def savemsg(self,msg):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        save_path=os.path.join(self.save_path,'{}.jpg'.format(msg["results"]["info"]["timestamp"]))
        img=msg["image"]
        if isinstance(img, str):
            # 解码base64字符串为NumPy数组
            nparr = np.frombuffer(base64.b64decode(img), dtype=np.uint8)
            # 使用OpenCV将NumPy数组转换为图像
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(save_path,img)
        self.logger.info("保存消息成功")
def convert_ndarray_to_list(data):
    """
    递归地遍历数据结构，将NumPy数组转换为列表。

    参数:
    data (dict or list or tuple or set or any iterable): 要遍历和转换的数据结构。

    返回:
    转换后的数据结构，其中NumPy数组已被替换为列表。
    """
    if isinstance(data, dict):
        # 如果data是字典，则递归地遍历其值
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple, set)):
        # 如果data是列表、元组或集合，则递归地遍历其元素
        return type(data)(convert_ndarray_to_list(item) for item in data)
    elif isinstance(data, np.ndarray):
        # 如果data是NumPy数组，则将其转换为列表
        return data.tolist()
    else:
        # 如果data不是上述类型之一，则直接返回它
        return data
