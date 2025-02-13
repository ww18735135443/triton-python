import argparse
import time

from model.model_infer.yolov8seg_triton_infer import YoloV8segTritonDetector
from model.model_infer.yolov5det_triton_infer import YoloV5TritonDetector
from model.model_infer.yolov8det_triton_infer import YoloV8TritonDetector
from flask import Flask, Response
import cv2
import numpy as np
from model.model_infer.tools.parser import get_config

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--videos",type=str,default='/home/ww/work/project/rtsp_opencv_demo-master/videos/20241028T090331Z_20241028T091331Z.mp4',help='videos path')
    # parser.add_argument("--videos",type=str,default='/home/ww/work/project/triton_project/data/org_videos/steel_concat.mp4',help='videos path')
    # parser.add_argument("--videos",type=str,default='https://open.ys7.com/v3/openlive/AC8525297_5_1.m3u8?expire=1761370468&id=771728734842818560&t=85ccb414ccaca43cf2ce673e3f7872acf89770a4688015c08593c0ba08efd782&ev=100',help='videos path')
    parser.add_argument("--videos",type=str,default='https://open.ys7.com/v3/openlive/AC8525297_6_1.m3u8?expire=1761371848&id=771734521816961024&t=3ccbedd3caf55e71147627cb32170c0a140681fde3ad1defe473598f7cb2e6f9&ev=100',help='videos path')

    parser.add_argument('--model_name',type=str,default='scsmodel',help='model name')
    parser.add_argument('--port',type=int,default=5000,help='port number')
    return parser.parse_args()

# 创建视频流生成器
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps==0:
        fps=25
    time_freq = 1 / fps
    i=0
    while cap.isOpened():
        i+=1
        # 构造一个图像帧
        rval, ori_img = cap.read()
        start_time = time.time()
        if i%5 !=0:
            # print(time_freq)
            time.sleep(time_freq)
            continue
        if opt.model_name=='scsmodel':

            boxes, segments, masks = tritondetector(ori_img)

            if len(boxes) > 0:
                frame=tritondetector.draw_detections(ori_img, boxes, segments, vis=False, save=True)
            else:
                frame=ori_img

        elif opt.model_name=='safetymodel' or opt.model_name=='steelcount':
            boxes,scores,classes = tritondetector(ori_img)
            if len(boxes) > 0:
                frame=tritondetector.draw_detections(ori_img,boxes,scores,classes)
            else:
                frame=ori_img
        end_time = time.time()
        print(end_time - start_time)
        # 使用cv2编码为JPEG格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 返回图像帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    print("open videos fail"    )
app = Flask(__name__)
@app.route('/video_feed')
def video_feed():
    # video_path='/home/ww/work/project/triton_project/data/org_videos/helmet.mp4'
    # video_path='/home/ww/work/project/rtsp_opencv_demo-master/videos/20241028T090331Z_20241028T091331Z.mp4'
    print(opt.videos)
    return Response(generate_frames(opt.videos),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



opt=parse_opt()

if opt.model_name == 'scsmodel':
    config_path='model/model_infer/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    tritondetector = YoloV8segTritonDetector(opt.model_name,cfg)
elif opt.model_name == 'safetymodel':
    config_path='model/model_infer/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    tritondetector = YoloV5TritonDetector(opt.model_name,cfg)
elif opt.model_name == 'steelcount':
    config_path='model/model_infer/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    tritondetector = YoloV8TritonDetector(opt.model_name,cfg)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=opt.port)
