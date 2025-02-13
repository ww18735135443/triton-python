import argparse

from yolov8seg_triton_infer import YoloV8TritonDetector
from flask import Flask, Response
import cv2
import numpy as np
from model.model_infer.tools.parser import get_config

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos",type=str,default='/home/ww/work/project/rtsp_opencv_demo-master/videos/20241028T090331Z_20241028T091331Z.mp4',help='videos path')
    parser.add_argument('--model_name',type=str,default='scsmodel',help='model name')
    parser.add_argument('--port',type=int,default=5000,help='port number')
    return parser.parse_args()

# 创建视频流生成器
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        # 构造一个图像帧
        rval, ori_img = cap.read()
        boxes, segments, masks = tritondetector(ori_img)
        if len(boxes) > 0:
            frame=tritondetector.draw_and_visualize(ori_img, boxes, segments, vis=False, save=True)
        else:
            frame=ori_img
        # 使用cv2编码为JPEG格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 返回图像帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
app = Flask(__name__)
@app.route('/video_feed')
def video_feed():
    # video_path='/home/ww/work/project/triton_project/data/org_videos/helmet.mp4'
    video_path='/home/ww/work/project/rtsp_opencv_demo-master/videos/20241028T090331Z_20241028T091331Z.mp4'
    return Response(generate_frames(opt.videos),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


config_path='/home/ww/work/project/triton_project/config/triton_config.yaml'
opt=parse_opt()
cfg = get_config()
cfg.merge_from_file(config_path)
tritondetector = YoloV8TritonDetector(opt.model_name,cfg)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
