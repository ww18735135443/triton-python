import numpy as np
from algrithom.tool.common import read_areas
from algrithom.tool.logic import WarnLogic
from algrithom.tool.logger import get_logger
from algrithom.tool.msgApp import msgApp
from model.model_infer.yolov8seg_triton_infer import YoloV8segTritonDetector
from model.model_infer.tools.parser import get_config
from algrithom.tool.draw import draw_areas,draw_detections
from shapely.geometry import Polygon,Point,LineString
import threading
import os
from model.trackers.bot_sort import BOTSORT
import cv2
from model.trackers.byte_tracker import BYTETracker
import base64
class WarnConfig:
    def __init__(self,param):
        self.alarm_last=param["alarm_last"]
        self.alarm_interval= param["alarm_interval"]
        self.alarm_classes=param["alarm_classes"]
        self.abnormalPercent=0.6
class AlgrithmLogic:
    _instance = None
    def __new__(cls, param):
        if not cls._instance:
            cls._instance = super(AlgrithmLogic, cls).__new__(cls)
            cls._initialized = False  # 初始化一个标志位来跟踪是否已经初始化
        return cls._instance

    def __init__(self, taskparam):
        if not self._initialized:
            self.warnConfig = WarnConfig(taskparam.config)
            self.Logic = self.logicInit()
            self.areas = read_areas(taskparam.areas)
            self._initialized = True  # 标记为已初始化
    def logicInit(self):
        logic = WarnLogic(self.warnConfig.alarm_last,self.warnConfig.alarm_interval,
                          self.warnConfig.abnormalPercent);
        return logic
    def algorithmLogic(self,detect,timestamp):
        Logic=self.Logic
        warnFlag=0
        warnFlag = Logic.update(detect, timestamp)
        return warnFlag
    def run(self,detections,timestamp):
        '''
        detections:[[x,y,x,y,track_id,conf,cls]] or [[x,y,x,y,conf,cls]]
        '''
        #检测类别存在性判断
        warn_object=[]
        if len(detections)<1:
            return 0,[]
        for detection in detections:
            cls=detection['cls']
            if cls in self.warnConfig.alarm_classes:
                #区域位置判断
                if len(self.areas)>0:
                    for i, area in enumerate(self.areas):
                        cur_xyxy=detection["xyxy"]
                        cur_bottom_center=Point((cur_xyxy[0] + cur_xyxy[2]) / 2, cur_xyxy[3])
                        if area.polygon.contains(cur_bottom_center):
                            warn_object.append(detection)
                else:
                    warn_object.append(detection)
        frameFlag=1 if len(warn_object)>0 else 0
        warnFlag=self.algorithmLogic(frameFlag,timestamp)
        return warnFlag,warn_object

TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}

def trace_data_preprocess(boxes,scores,cls):
    detections = {
        'xyxy': np.array(boxes),  # 边界框坐标
        'conf': np.array(scores),  # 置信度
        'cls': np.array(cls),  # 类别ID
        # 这里可以添加YOLOv8特有的其他字段，如果需要的话
    }
    return detections

def trace_data_postprocess(results,model_cls_list):
    detections=[]
    for result in results:
        detection = {
            'xyxy': np.array(result[:4]),  # 边界框坐标
            'conf': np.array(result[4]),  # 置信度
            'cls': model_cls_list[int(np.array(result[5]))],  # 类别ID
            'track_id': int(np.array(result[6]))
            # 这里可以添加YOLOv8特有的其他字段，如果需要的话
        }
        detections.append(detection)
    return detections
# 定义一个回调函数

def result_process(warn_detect,frame,param,warn_flag,timestamp):
    msg={
        "camera_address": "",
        "interfaceId":"",
        "algorithm_type":"", #算法类型
        "results":{
            "info": {
                "timestamp": "<dateTime data>",
                "Event_type": ""  #事件类型,未检测出类型则为None
            },
            "data": [] #返回检测框，置信度，类别，跟踪id，是否为触发告警目标。
        }
        # "image":"imgbase64"
    }

    msg['camera_address']=param['videosTask']['videosId']
    msg['interfaceId']=param['interfaceId']
    msg['algorithm_type']=param['algorithmType']
    msg["results"]["info"]["timestamp"]=timestamp
    msg["results"]["info"]["Event_type"]=param['videosTask']['config']['alarm_classes']
    msg["results"]["data"]=warn_detect
    # msg={}
    # msg["image"]=base64.b64encode(frame).decode('utf-8')
    msg["image"]=frame
    msg["warnflag"]=warn_flag
    return msg
class MechineAlgThread(threading.Thread):
    def __init__(self,param):
        threading.Thread.__init__(self)
        self.param = param
        """初始化路径、日志"""

        self.save_log_path = os.path.join('data/logs', "alg_log.txt")
        self.logger = get_logger(self.save_log_path)
        self.logger.info('*' * 50)
        self.logger.info(param)
        """加载参数"""
        # self.config_path = param["configPath"]
        self.queue = param["pictureQueue"]
        self.camera_id = param["videosTask"].videosId
        self.areas=read_areas(param["videosTask"].areas)
        self.alarm_config = param["videosTask"]
        self.detect_cfg=param['triton_cfg']
        self.tracker_cfg=param['tracker_cfg']
        self.logger.info("tracker param:{}".format(self.tracker_cfg))
        self.model_cls=param['triton_cfg']['model_info'][param.model_name]['labels']
        """初始化模型、算法"""
        if 'kafka' in param and 'kafkaIP' in param['kafka'] and param['kafka']['kafkaIP'] != []:
            self.msgapp=msgApp(bootstrap_servers = param['kafka']['kafkaIP'])
            self.msgapp.register_callback(self.sendkafkamsg)
        else:
            self.logger.info("kafka param is empty")
            self.msgapp=msgApp()
        if 'save' in param and 'path' in param['save'] and isinstance(param['save']['path'],str):
            # self.msgapp.save_path=os.path.join(param['save']['path'],param['topic'])
            self.save_path=os.path.join(param['save']['path'],param['topic'])
            self.msgapp.register_callback(self.savemsg)
        self.detector =YoloV8segTritonDetector(param.model_name,self.detect_cfg)
        self.logger.info('检测模型加载成功')
        self.trackmodel = TRACKER_MAP[param["trackerType"]](self.tracker_cfg,frame_rate=30)
        self.logger.info('安全帽佩戴检测算法线程初始化成功！')
        self.logic=AlgrithmLogic(self.alarm_config)
        targets = [self.run]
        for target in targets:
            curThread = threading.Thread(target=target, args=(), daemon=True)
            curThread.start()
        self.logger.info('安全帽佩戴检测算法线程启动成功')
    def sendkafkamsg(self,msg):
        if self.msgapp.kafka_send:
            msg["image"]=base64.b64encode(msg["image"]).decode('utf-8')
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
        cv2.imwrite(save_path,msg["image"])
        self.logger.info("保存消息成功")


    def run(self):
        while True:
            content = self.queue.get()
            frame = content['picture']
            timestamp = content['timestamp']
            # imagepath = '/home/ww/work/project/triton_project/4.jpg'
            # frame = cv2.imread(imagepath)
            boxes,scores,cls = self.detector(frame)
            detections=trace_data_preprocess(boxes,scores,cls)
            track_result=self.trackmodel.update(detections)
            track_result=trace_data_postprocess(track_result,self.model_cls)
            # print(track_result)

            warn_flag, warn_object = self.logic.run(track_result, timestamp)
            self.logger.info("warn_flag:{}, timestamp:{},warn_object:{}".format(warn_flag,timestamp,warn_object))
            if  warn_flag:
                draw_areas(frame, self.areas)  # 画区域
                draw_detections(frame,warn_object)
                # cv2.imshow("warn fig",frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                msg = result_process(warn_object,frame, self.param,warn_flag, timestamp)
                self.msgapp.send(msg)


