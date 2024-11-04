from kafka import KafkaProducer
import json
class KafkaApp:
    def __init__(self, bootstrap_servers=None, logger=None):
        self.bootstrap_servers = bootstrap_servers
        # self.bootstrap_servers = ['10.5.56.172:9092', '10.5.56.79:9092', '10.5.56.174:9092']

        self.producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers,
                                      value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                      key_serializer=lambda v: json.dumps(v).encode('utf-8'))
    def register_callback(self, callback):
        """注册回调函数"""
        if callable(callback):
            self._callbacks.append(callback)
        else:
            raise ValueError("提供的不是一个可调用对象")

    def send(self, msg):

        """发送事件"""

        print("发送kafka消息！")

        for callback in self._callbacks:
            callback( msg)