# -*-coding:GBK -*-
import time
import queue
import threading
import argparse
from algrithom import *
from model.model_infer.tools.parser import get_config,load_config_from_file
import cv2
import os

fps=25
def decode(srcQueue, url):
    cap = cv2.VideoCapture()
    cap.open(url)
    global fps
    fps= cap.get(cv2.CAP_PROP_FPS)

    count = 1
    while cap.grab():
        _, ori_im = cap.retrieve()
        if count % 5 != 0:
            count += 1
            continue
        srcQueue.put(ori_im)



def vas(srcQueue, picQueue):
    cam_id = ''
    timestamp = time.time()
    time_freq = 1 / fps
    count=0
    while True:
        count += 1
        timestamp += time_freq
        frame = srcQueue.get()
        msg_format = {"picture": frame, "camera_id": cam_id, "timestamp": timestamp, "frame_number": count}
        picQueue.put(msg_format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config/helmet_algo_config.yaml', help='config path')
    # parser.add_argument('--algtype', type=str, default='yanhuodetect', help='')
    opt = parser.parse_args()
    paramdic = get_config()
    paramdic.merge_from_file(opt.cfg)
    triton_config_path='model/model_infer/config/triton_config.yaml'
    triton_cfg = get_config()
    triton_cfg.merge_from_file(triton_config_path)
    paramdic['triton_cfg'] = triton_cfg
    if paramdic['trackerType']=='bytetrack':
        tracker_cfg_path='model/trackers/cfg/bytetrack.yaml'
    else:
        tracker_cfg_path='model/trackers/cfg/botsort.yaml'
    paramdic['tracker_cfg'] = load_config_from_file(tracker_cfg_path)
    url = paramdic.videosTask.videosId
    srcQueue = queue.Queue(maxsize=30)
    picQueue = queue.Queue(maxsize=50)
    paramdic['pictureQueue'] = picQueue

    if paramdic.algorithmType == 'smokefiredetect':
        paramdic["model_name"] = 'smokefiremodel'
        algorithm = SmokefireAlgThread
    elif paramdic.algorithmType == 'fencedetect':
        paramdic["model_name"] = 'fencemodel'
        algorithm = FenceAlgThread
    elif paramdic.algorithmType=='helmetdetect':
        paramdic["model_name"] = 'wearmodel'
        algorithm = HelmetAlgThread
    elif paramdic.algorithmType=='safebeltdetect':
        paramdic["model_name"] = 'wearmodel'
        algorithm = SafebeltAlgThread
    elif paramdic.algorithmType=='vasdetect':
        paramdic["model_name"] = 'scsmodel'
        algorithm = VasAlgThread
    elif paramdic.algorithmType=='crosslinedetect':
        paramdic["model_name"] = 'scsmodel'
        algorithm = CrosslineAlgThread
    elif paramdic.algorithmType=='mechinedetect':
        paramdic["model_name"] = 'scsmodel'
        algorithm = MechineAlgThread
    elif paramdic.algorithmType=='crowdcountdetect':
        paramdic["model_name"] = 'scsmodel'
        algorithm = CrowdcountAlgThread
    elif paramdic.algorithmType=='reflectivevestdetect':
        paramdic["model_name"] = 'wearmodel'
        algorithm = FlectivevestAlgThread
    else:
        print('算法类型字段错误')
    resQueue = queue.Queue(maxsize=500)
    paramdic['resultQueue'] = resQueue
    decodeThread = threading.Thread(target=decode, args=(srcQueue, url))
    decodeThread.start()
    vasThread = threading.Thread(target=vas, args=(srcQueue, picQueue,))
    vasThread.start()
    retentionThread = threading.Thread(target=algorithm, args=(paramdic,))
    retentionThread.start()
