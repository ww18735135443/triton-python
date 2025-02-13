import numpy as np
from algrithom.tool.common import read_areas,save_data
from algrithom.tool.logic import WarnLogic
from algrithom.tool.logger import get_logger
from algrithom.tool.msgApp import msgApp
from algrithom.tool.algthread import AlgThread
from model.model_infer.yolov8seg_triton_infer import YoloV8segTritonDetector
from model.model_infer.tools.parser import get_config
from algrithom.tool.draw import draw_areas,draw_detections
from shapely.geometry import Polygon,Point,LineString
import threading
import os
from model.trackers.bot_sort import BOTSORT
import queue
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
        warnFlag=0
        warnFlag = self.Logic.update(detect, timestamp)
        return warnFlag
    def run(self,detections,timestamp):
        '''
        detections:[[x,y,x,y,track_id,conf,cls]] or [[x,y,x,y,conf,cls]]
        '''
        #检测类别存在性判断
        warn_object=[]
        if len(detections)<1:
            return 0,[]
        count=0
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
                            area.count+=1
                            count+=1
                else:
                    warn_object.append(detection)
                    count+=1
        frameFlag=1 if len(warn_object)>0 else 0
        warnFlag=self.algorithmLogic(frameFlag,timestamp)
        result= {}
        if warnFlag:
            if len(self.areas)>0:
                statistic_result ={"statistic_result":[{"region": area.region_index, "num": area.count} for i, area in enumerate(self.areas)]}
            else:
                statistic_result={"statistic_result":count}
            result={
                "statistic_result":statistic_result,
                "detect_result":warn_object
            }
        for area in self.areas:
            area.count=0
        return warnFlag,result


TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}

def trace_data_preprocess(detections):
    if len(detections)==0:
        detections = {
            'xyxy': np.array([]),  # 边界框坐标
            'conf': np.array([]),  # 置信度
            'cls': np.array([]),  # 类别ID
            # 这里可以添加YOLOv8特有的其他字段，如果需要的话
        }
    else:
        detections = {
            'xyxy': np.array(detections[:,:4]),  # 边界框坐标
            'conf': np.array(detections[:,4]),  # 置信度
            'cls': np.array(detections[:,5]),  # 类别ID
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

class CrowdcountAlgThread(threading.Thread,AlgThread):
    def __init__(self,param):
        threading.Thread.__init__(self)
        self.param = param
        """初始化路径、日志"""

        self.save_log_path = os.path.join('data/logs', "alg_log.txt")
        self.logger = get_logger(self.save_log_path,param['interfaceId'])
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
        self.logger.info('人群计数算法线程初始化成功！')
        self.logic=AlgrithmLogic(self.alarm_config)

        targets = [self.run]
        for target in targets:
            curThread = threading.Thread(target=target, args=(), daemon=True)
            curThread.start()
        self.logger.info('人群计数算法线程启动成功')


    def run(self):
        while True:
            try:
                content = self.queue.get(block=True, timeout=1)
                frame = content['picture']
                timestamp = content['timestamp']
                # imagepath = '/home/ww/work/project/triton_project/157368844_23.jpg'
                # frame = cv2.imread(imagepath)
                detections, segments, masks = self.detector(frame)
                detections=trace_data_preprocess(detections)
                track_result=self.trackmodel.update(detections)
                track_result=trace_data_postprocess(track_result,self.model_cls)
                warn_flag,warn_msg = self.logic.run(track_result, timestamp)
                # print(timestamp)
                # cv2.imwrite("/home/ww/work/project/triton_project/data/test_data/{}.jpg".format(timestamp),frame)
                self.logger.info("person statistic result:{}".format(warn_msg))
                if  warn_flag:
                    save_data(self.param['algorithmType'],frame,timestamp)
                    self.logger.info("warn_flag:{}, timestamp:{},warn_msg:{}".format(warn_flag,timestamp,warn_msg))
                    draw_areas(frame, self.areas)  # 画区域
                    draw_detections(frame,warn_msg['detect_result'])
                    # cv2.imshow("warn fig",frame)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    msg = result_process(warn_msg,frame, self.param,warn_flag, timestamp)
                    self.msgapp.send(msg)
            except queue.Empty:
                self.logger.info("picQueue已空，等待数据加入。")
                # 如果队列为空且没有设置停止事件，则继续循环
                continue
            except Exception as e:
                self.logger.info("算法处理过程中发生错误: %s", e)



