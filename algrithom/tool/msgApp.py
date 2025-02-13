import os
from algrithom.tool.logger import get_logger
from kafka import KafkaProducer
import json
import threading
from algrithom.tool.qwen_detect import qwen_detect
from algrithom.tool.common import img_2_base64
class msgApp:
    def __init__(self, bootstrap_servers=None, savepath=None):
        self.save_log_path = os.path.join('data/logs', "alg_log.txt")
        self.logger = get_logger(self.save_log_path)
        self.logger.info('*' * 50)
        self.savepath=savepath
        self._callbacks=[]
        self.kafka_send=0
        if bootstrap_servers:
            try:
                self.bootstrap_servers = bootstrap_servers
                # self.bootstrap_servers = ['10.5.56.172:9092', '10.5.56.79:9092', '10.5.56.174:9092']

                self.producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers,
                                          value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                          key_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                              max_request_size=100 * 1024 * 1024)
                self.kafka_send=1
                print('kafka init success')
                self.logger.info('kafka init success:{}'.format(bootstrap_servers))
            except Exception as e:
                print('kafka init fail')
                print(e)
                self.logger.info('kafka init fail'.format(bootstrap_servers))
        if savepath:
            self.savepath = savepath
            if not os.path.exists(self.savepath):
                os.makedirs(self.savepath)
    def register_callback(self, callback):
        """注册回调函数"""
        if callable(callback):
            self._callbacks.append(callback)
        else:
            raise ValueError("提供的不是一个可调用对象")

    def send(self, msg):

        """发送事件"""

        print("发送消息！")

        for callback in reversed(self._callbacks):
            callback( msg)
    def send(self, msg,vl_check=0):
        #是否需要调用大模型对报警事件进行检查
        if vl_check:
            try:
                self.logger.info('start vl_check,alg type:{}'.format(vl_check))
                if vl_check == 'helmetdetect':
                    content="图中有个蓝色的框，框的左上角有个英文单词，表示框内发生了该英文含义的事件的发生,请根据图片实际情况回答图中框内是否发生了该事件，如果是，请回答True，否则回答False"
                elif vl_check == 'safebeltdetect':
                    content="图中有个蓝色的框框选了一个人，表示框中的人没有系带安全带或安全绳,请根据图片实际情况回答图中蓝色框内的人是否没系安全带或安全绳，如果是，请回答True，否则回答False"
                elif vl_check == 'smokefiredetect':
                    content="图中有个蓝色的框，表示框中的存在烟火或烟雾,请根据图片实际情况回答图中蓝色框内的人是否存在烟火或烟雾，如果是，请回答True，否则回答False"
                elif vl_check == 'reflectivevestdetect':
                    content="图中有个蓝色的框框选了一个人，表示框中的人没有穿反光衣或者反光背心,请根据图片实际情况回答图中蓝色框内的人是否没穿反光衣或者反光背心，如果是，请回答True，否则回答False"
                elif vl_check == 'lifejacketdetect':
                    content="图中有个蓝色的框框选了一个人，表示框中的人没有穿救生衣,请根据图片实际情况回答图中蓝色框内的人是否没穿救生衣，如果是，请回答True，否则回答False"
                image=img_2_base64(msg['image'])
                vl_check_result=qwen_detect(content,image)
                if vl_check_result == 0:
                    self.logger.info('vl_check fail,alg type:{}'.format(vl_check))
                    #视觉大模型检查未通过,不发送报警消息
                    return 0
                self.logger.info('vl_check success,alg type:{}'.format(vl_check))
            except Exception as e:
                self.logger.info('vl_check fail,{}'.format(e))
        """发送事件"""
        for callback in reversed(self._callbacks):
            callback( msg)