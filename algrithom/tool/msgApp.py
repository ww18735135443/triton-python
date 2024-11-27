import os

from kafka import KafkaProducer
import json
class msgApp:
    def __init__(self, bootstrap_servers=None, savepath=None):
        self.savepath=None
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
            except Exception as e:
                print('kafka init fail')
                print(e)
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