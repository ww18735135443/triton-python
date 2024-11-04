import numpy as np
from algrithom.tool.common import read_areas
from algrithom.tool.logic import WarnLogic
from algrithom.tool.logger import get_logger
from algrithom.tool.kafka import KafkaApp
from model.model_infer.yolov5det_triton_infer import YoloV5TritonDetector
from model.model_infer.tools.parser import get_config
from shapely.geometry import Polygon,Point,LineString
import threading
import os
from model.trackers.bot_sort import BOTSORT
import cv2
from model.trackers.byte_tracker import BYTETracker
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
        if len(detections)==1 and "nothing" in detections[0].keys():
            return 0,[]
        for detection in detections:
            cls=detection["cls"]
            if cls in self.warnConfig.warn_class:
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
# 定义一个回调函数

def result_process(warn_detect,param,warn_flag,timestamp):
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

    msg['camera_address']=param['camera_address']
    msg['interfaceId']=param['interfaceId']
    msg['algorithm_type']=param['algorithm_type']
    msg["results"]["info"]["timestamp"]=timestamp
    msg["results"]["info"]["Event_type"]=param["warn_class"]
    msg["results"]["data"]=warn_detect
    # msg={}
    msg["image"]=param["image"]
    msg["warnflag"]=warn_flag
    return msg
class YanhuoAlgThread(threading.Thread):
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
        self.areas=param["videosTask"].areas
        self.alarm_config = param["videosTask"]
        self.detect_cfg=param['triton_cfg']
        self.tracker_cfg=param['tracker_cfg']
        """初始化模型、算法"""
        if 'kafka' in param and 'kafkaIP' in param['kafka'] and param['kafka']['kafkaIP'] != []:
            self.Kafkapp=KafkaApp(bootstrap_servers = param['kafkaIP'])
            self.Kafkapp.register_callback(self.sendkafkamsg)
        if 'save' in param and 'path' in param['save'] and isinstance(param['save']['path'],str):
            self.save_path=os.path.join(param['save']['path'],param['topic'])
            self.Kafkapp.register_callback(self.savemsg)
        self.detector =YoloV5TritonDetector(param.model_name,self.detect_cfg)
        self.logger.info('检测模型加载成功')
        self.trackmodel = TRACKER_MAP[param["trackerType"]](self.tracker_cfg,frame_rate=30)
        self.logger.info('入侵算法线程初始化成功！')
        self.logic=AlgrithmLogic(self.alarm_config)
        targets = [self.run]
        for target in targets:
            curThread = threading.Thread(target=target, args=(), daemon=True)
            curThread.start()
        self.logger.info('入侵线程启动成功')
    def sendkafkamsg(self,msg):
        pass

        print("发送消息成功")
    def savemsg(self,msg):
        pass

    def run(self):
        while True:
            content = self.queue.get()
            frame = content['picture']
            timestamp = content['timestamp']
            imagepath = '/home/ww/work/project/triton_project/4.jpg'
            frame = cv2.imread(imagepath)
            boxes,scores,cls = self.detector(frame)
            detections=trace_data_preprocess(boxes,scores,cls)
            track_result=self.trackmodel.update(detections)
            print(track_result)
            warn_flag, warn_object = self.logic.run(track_result, timestamp)
            print(warn_flag, timestamp)
            if  warn_flag:
                bboxs, scores, class_ids=warn_object
                warn_fig=self.detector.draw_detections(frame,bboxs, scores, class_ids)
                msg = result_process(warn_object, self.param,warn_flag, timestamp)
                self.Kafkapp.send(msg)



