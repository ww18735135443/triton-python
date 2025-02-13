import numpy as np
import time
from model.trackers.bot_sort import BOTSORT
from model.trackers.byte_tracker import BYTETracker

import json
import logging

# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}


import yaml
def load_config_from_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

class yolov5_track:
    def __init__(self,cfg):
        self.cfg = cfg
        self.tracker = TRACKER_MAP[cfg["tracker_type"]](load_config_from_file("./cfg/" + cfg["tracker_type"] + ".yaml"),frame_rate=30)
        self.trackers_result = None
    def track(self,pre):
        self.trackers_result =self.tracker.update(pre)
        return self.trackers_result
class track_demo:
    def __init__(self):
        self.track=yolov5_track(cfg)
    def convert_yolov5_to_yolov8(self,yolov5_outputs):
        if not isinstance(yolov5_outputs, np.ndarray):
            # 如果不是NumPy数组，则尝试将其转换为NumPy数组
            # 注意：这里假设输入是可迭代的（如列表、元组等）
            try:
                yolov5_outputs = np.array(yolov5_outputs)
            except Exception as e:
                print(f"无法将输入转换为NumPy数组: {e}")
                return None  # 或者可以抛出异常，取决于你的需求
        yolov8_detection = {
            'xyxy': yolov5_outputs[:,:4],  # 边界框坐标
            'conf': yolov5_outputs[:,4],  # 置信度
            'cls': yolov5_outputs[:,5],  # 类别ID
            # 这里可以添加YOLOv8特有的其他字段，如果需要的话
        }

        return yolov8_detection
    def run(self,pres):

        results=self.convert_yolov5_to_yolov8(pres)
        # results=pres[0]
        track_result=self.track.track(results)
        return track_result
app = Flask(__name__)
# 获取 Flask 应用的 logger
# 获取 Flask 应用的 logger
app_logger = app.logger  # 使用 app_logger 避免与 logging.getLogger(__name__) 混淆

# 配置 Flask 日志（注意这里我们不使用 logging.basicConfig()，因为它会影响全局的 logging 配置）
# 你可以为 app.logger 添加一个 StreamHandler 来输出到控制台
app_logger.addHandler(logging.StreamHandler())
# 设置日志级别为 DEBUG
app_logger.setLevel(logging.DEBUG)
track_type = {
    "tracker_type": "bytetrack",
}
cfg=load_config_from_file("./trackers/cfg/" + track_type["tracker_type"] + ".yaml")
track_test=track_demo()
@app.route("/tracks/v1", methods=["POST"])
def process():
    try:
        start_time = time.time()
        data = request.get_json()
        detections = data["detections"]
        response_result=track_test.run(detections)
        end_time = time.time()
        app_logger.debug("detections tracks time:{:6f}".format(end_time - start_time))
        return_dict = jsonify(response_result.tolist())
    except Exception as e:
        print(e)
        return_dict = json.dumps({"code": -1, "msg": "fail", "data": []})
    return return_dict


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8296, debug=False)
    # track_type = {
    #     "tracker_type": "botsort",
    # }
    # cfg=load_config_from_file("./trackers/cfg/" + track_type["tracker_type"] + ".yaml")
    # track_test=track_demo()
    #
    # for i in range(100):
    #     track_test.run("/mnt/zj/datasets/steel_count/test/fa31cea13894704836ccbf98f141c3f2.jpg")










