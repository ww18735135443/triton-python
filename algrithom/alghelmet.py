from algrithom.tool.common import read_areas
from algrithom.tool.logic import WarnLogic
from algrithom.tool.logger import get_logger
from model.model_infer.yolov5det_triton_infer import YoloV5TritonDetector
from model.model_infer.tools.parser import get_config
from shapely.geometry import Polygon,Point,LineString
import threading
import os
from model.trackers.bot_sort import BOTSORT
from model.trackers.byte_tracker import BYTETracker
class WarnConfig:
    def __init__(self,param):
        self.warnThreshold=param["warnThreshold"]
        self.alarm_interval= param["alarm_interval"]
        self.abnormalPercent=0.6
class AlgrithmLogic:
    _instance = None
    def __new__(cls, param):
        if not cls._instance:
            cls._instance = super(AlgrithmLogic, cls).__new__(cls)
            cls._initialized = False  # 初始化一个标志位来跟踪是否已经初始化
        return cls._instance

    def __init__(self, param):
        if not self._initialized:
            self.warnConfig = WarnConfig(param)
            self.Logic = self.logicInit()
            self.warn_class = param["warn_class"]
            self.areas = read_areas(param)
            self._initialized = True  # 标记为已初始化
    def logicInit(self):
        logic = WarnLogic(self.warnConfig.warnThreshold,self.warnConfig.alarm_interval,
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
            cls=detection["classes"]
            if cls in self.warn_class:
                #区域位置判断
                if len(self.areas)>0:
                    for i, area in enumerate(self.areas):
                        cur_xyxy=detection["box"]
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
        'xyxy': boxes,  # 边界框坐标
        'conf': scores,  # 置信度
        'cls': cls,  # 类别ID
        # 这里可以添加YOLOv8特有的其他字段，如果需要的话
    }
    return detections
class YanhuoAlgThread(threading.Thread):
    def __init__(self,param):
        threading.Thread.__init__(self)
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
        self.alarm_config = param["videosTask"].config
        self.detect_cfg=param['triton_cfg']
        self.tracker_cfg=param['tracker_cfg']
        """初始化模型、算法"""
        self.detector =YoloV5TritonDetector(param.model_name,self.detect_cfg)
        self.logger.info('检测模型加载成功')
        self.trackmodel = TRACKER_MAP[param["trackerType"]](self.tracker_cfg,frame_rate=30)
        self.logger.info('入侵算法线程初始化成功！')
        # video_save_path = self.save_path + '/area_result.mp4'
        # print("视频保存路径:",video_save_path)
        # self.writer = imageio.get_writer(video_save_path, fps=25)
        targets = [self.inference]
        for target in targets:
            curThread = threading.Thread(target=target, args=(), daemon=True)
            curThread.start()
        self.logger.info('入侵线程启动成功')
    def inference(self):
        while True:
            content = self.queue.get()
            frame = content['picture']
            boxes,scores,cls = self.detector(frame)
            detections=trace_data_preprocess(boxes,scores,cls)
            track_result=self.trackmodel.update(detections)
            print(track_result)

