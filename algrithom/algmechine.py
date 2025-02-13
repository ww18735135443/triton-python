import numpy as np
from algrithom.tool.common import read_areas,ResultProcess,save_data
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
        Logic=self.Logic
        warnFlag=0
        warnFlag = Logic.update(detect, timestamp)
        return warnFlag
    def run(self,detections,statustrack,timestamp):
        '''
        detections:[[x,y,x,y,track_id,conf,cls]] or [[x,y,x,y,conf,cls]]
        '''

        if len(detections)<1:
            return []

        msg = {
            'detection_object': detections,  # 检测结果
            'statics': statustrack.state,  # 统计结果
            'changs_object': statustrack.object_changes,  # 类别变化
            # 这里可以添加YOLOv8特有的其他字段，如果需要的话
        }
        return msg

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


class TrackerStatus:
    _instance = None
    def __new__(cls, alarm_config):
        if not cls._instance:
            cls._instance = super(TrackerStatus, cls).__new__(cls)
            cls._initialized = False  # 初始化一个标志位来跟踪是否已经初始化
        return cls._instance
    def __init__(self, alarm_config):
        if not self._initialized:
            self.classes = alarm_config.config.alarm_classes
            self.state = {cls: {"count": 0, "track_ids": set()} for cls in self.classes}
            self._initialized = True  # 标记为已初始化
            self.areas = read_areas(alarm_config.areas)
            self.object_changes=[]

    def update(self, detections):
        if len(detections) < 1:
            return []
        new_state = {cls: {"count": 0, "track_ids": set()} for cls in self.classes}
        # 更新新状态
        for detection in detections:
            if "cls" in detection and detection["cls"] in self.classes:
                if len(self.areas)>0:
                    for i, area in enumerate(self.areas):
                        cur_xyxy=detection["xyxy"]
                        cur_bottom_center=Point((cur_xyxy[0] + cur_xyxy[2]) / 2, cur_xyxy[3])
                        if area.polygon.contains(cur_bottom_center):
                            class_name = detection["cls"]
                            new_state[class_name]["count"] += 1
                            if "track_id" in detection:
                                new_state[class_name]["track_ids"].add(detection["track_id"])
                else:
                    class_name = detection["cls"]
                    new_state[class_name]["count"] += 1
                    if "track_id" in detection:
                        new_state[class_name]["track_ids"].add(detection["track_id"])
                # 分析变化
        changes = []
        for cls in self.classes:
            old_count = self.state[cls]["count"]
            old_ids = set(self.state[cls]["track_ids"])
            new_count = new_state[cls]["count"]
            new_ids = new_state[cls]["track_ids"]

            # 检查数量变化
            add_id=new_ids - old_ids
            for idx in add_id:
                change_object={}
                change_object["cls"] =cls
                change_object["track_id"] = idx
                changes.append(change_object)
        self.object_changes=changes
        # 更新状态

        self.state = new_state
        return changes
class MechineAlgThread(threading.Thread,AlgThread):
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
        self.logger.info('机械车辆识别算法线程初始化成功！')
        self.logic=AlgrithmLogic(self.alarm_config)
        self.statustrack=TrackerStatus(self.alarm_config)
        targets = [self.run]
        for target in targets:
            curThread = threading.Thread(target=target, args=(), daemon=True)
            curThread.start()
        self.logger.info('机械车辆识别算法线程启动成功')
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
                object_changes=self.statustrack.update(track_result)
                warn_flag=1 if len(object_changes)>0 else 0
                # self.logger.info("mechine change result:{}".format(warn_flag))
                if  warn_flag:
                    save_data(self.param['algorithmType'],frame,timestamp)
                    warn_msg = self.logic.run(track_result,self.statustrack, timestamp)
                    self.logger.info("warn_flag:{}, timestamp:{},warn_object:{}".format(warn_flag,timestamp,warn_msg))
                    draw_areas(frame, self.areas)  # 画区域
                    draw_detections(frame,warn_msg['detection_object'])
                    # cv2.imshow("warn fig",frame)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    msg = ResultProcess.mechine_result_process(warn_msg,frame, self.param,warn_flag, timestamp)
                    threading.Thread(target=lambda: self.send_message(msg), daemon=True).start()
            except queue.Empty:
                self.logger.info("picQueue已空，等待数据加入。")
                # 如果队列为空且没有设置停止事件，则继续循环
                continue
            except Exception as e:
                self.logger.info("算法处理过程中发生错误: %s", e)



