import numpy as np
from collections import deque
from algrithom.tool.logic import WarnLogic
def calculate_ioua(boxa, boxb):
    """
    计算a与b的交集除a的面积,即a与b重叠的部分占a的面积比例
    :param boxa:
    :param boxb:
    :return:
    """
    # 确保输入是 numpy 数组
    boxa = np.asarray(boxa, dtype=np.float32)
    boxb = np.asarray(boxb, dtype=np.float32)
    # 计算交集区域的坐标
    xi1 = np.maximum(boxa[0], boxb[0])
    yi1 = np.maximum(boxa[1], boxb[1])
    xi2 = np.minimum(boxa[2], boxb[2])
    yi2 = np.minimum(boxa[3], boxb[3])

    # 计算交集区域的宽度和高度
    inter_width = np.maximum(0.0, xi2 - xi1)

    inter_height = np.maximum(0.0, yi2 - yi1)
    # 计算交集区域的面积
    inter_area = inter_width * inter_height
    # 计算 boxa 的面积
    boxa_area = (boxa[2] - boxa[0]) * (boxa[3] - boxa[1])
    # 计算 IOUA
    ioua = inter_area / boxa_area if boxa_area > 0 else 0.0
    return ioua

class Tracker:
    def __init__(self,detection):
        self.current_xyxy=detection['xyxy']
        self.track_id=detection['track_id']
        self.classes=detection['cls']
        self.conf=detection['conf']
        self.alarm_time=None
        self.alarm_state=None
        self.last_xyxy=detection['xyxy']
        self.age=0
        self.attrThreshold=0.9 #ioua的阈值
        self.wearattr={'helmet':2,'belt':2,'reflectivevest':2,'lifejacket':2}
        self.helmet=deque()
        self.belt=deque()
        self.reflectivevest=deque()
        self.lifejacket=deque()
    def update(self,detection):
        self.last_xyxy =self.current_xyxy
        self.current_xyxy=detection['xyxy']
        self.age=0

class Tracks:
    def __init__(self):
        self.tracks=[]
        self.track_id=[]
    def update(self,detections):
        self.track_id = [track.track_id for track in self.tracks]
        for detection in detections:
            if detection['track_id'] not in self.track_id:
                self.tracks.append(Tracker(detection))
            else:
                for track in self.tracks:
                    if track.track_id==detection['track_id']:
                        track.update(detection)
        for track in self.tracks:
            track.age+=1
        tracks=[track for track in self.tracks if track.age<=60]
        self.tracks=tracks
    def wearupdate(self,detections,wearattr='belt'):
        if wearattr=='helmet':
            for track in self.tracks:
                is_wear=0
                for detection in detections:
                    if detection['cls']=='helmet':
                        wear_xyxy=detection['xyxy']
                        ioua=calculate_ioua(wear_xyxy, track.current_xyxy)
                        if ioua>track.attrThreshold:
                            track.helmet.appendleft(1)
                            is_wear=1
                            break
                if is_wear==0:
                    track.helmet.appendleft(0)
                track.wearattr[wearattr]=1 if sum(i>0 for i in track.helmet)>0 else 0
                while len(track.helmet)>25:
                    track.helmet.pop()
        if wearattr=='belt':
            for track in self.tracks:
                is_wear=0
                for detection in detections:
                    if detection['cls']=='belt':
                        wear_xyxy=detection['xyxy']
                        ioua=calculate_ioua(wear_xyxy, track.current_xyxy)
                        if ioua>track.attrThreshold:
                            track.belt.appendleft(1)
                            is_wear=1
                            break
                if is_wear==0:
                    track.belt.appendleft(0)
                if sum(i for i in track.belt)/len(track.belt)<0.1 and len(track.belt)>10 and track.wearattr[wearattr]==2:
                    track.wearattr[wearattr]=0
                elif sum(i for i in track.belt)/len(track.belt)>0.5:
                    track.wearattr[wearattr]=1
                while len(track.belt)>25:
                    track.belt.pop()
        if wearattr=='reflectivevest':
            for track in self.tracks:
                is_wear=0
                for detection in detections:
                    if detection['cls']=='reflectivevest':
                        wear_xyxy=detection['xyxy']
                        ioua=calculate_ioua(wear_xyxy, track.current_xyxy)
                        if ioua>track.attrThreshold:
                            track.reflectivevest.appendleft(1)
                            is_wear=1
                            break
                if is_wear==0:
                    track.reflectivevest.appendleft(0)
                if sum(i for i in track.reflectivevest)/len(track.reflectivevest)<0.1 and len(track.reflectivevest)>10:
                    track.wearattr[wearattr]=0
                else:
                    track.wearattr[wearattr]=2
                while len(track.reflectivevest)>25:
                    track.reflectivevest.pop()