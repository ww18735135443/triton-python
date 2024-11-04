# -*-coding:GBK -*-
import time
import queue
import threading
import argparse
from algrithom import *
from model.model_infer.tools.parser import get_config,load_config_from_file
import cv2
import os


def decode(srcQueue, url):
    cap = cv2.VideoCapture()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.open(url)
    i = 0
    while cap.grab():
        _, ori_im = cap.retrieve()
        srcQueue.put(ori_im)
        i += 1
        # print(i)


def vas(srcQueue, picQueue):
    cam_id = ''
    count = 1
    fps = 25
    timestamp = time.time()
    start_timestamp = timestamp
    time_freq = 1 / fps
    while True:
        timestamp += time_freq
        frame = srcQueue.get()
        # if count % 5 != 0:
        #     count += 1
        #     continue
        msg_format = {"picture": frame, "camera_id": cam_id, "timestamp": timestamp, "frame_number": count}
        # if int(timestamp - start_timestamp) % 10 == 0 and count % 25 == 0:
        #     print('passed {} seconds'.format(int(timestamp - start_timestamp)))
        count += 1
        picQueue.put(msg_format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config/algo_config.yaml', help='config path')
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

    if paramdic.algorithmType == 'yanhuodetect':
        paramdic["model_name"] = 'yanhuomodel'
        algorithm = YanhuoAlgThread
    # elif opt.type=='helmetdetect':
    #     paramdic["algorithmId"] = '49'
    #     algorithm = AreainvasionThread
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
