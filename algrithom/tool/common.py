import numpy as np
from shapely.geometry import Polygon,Point,LineString

class OneArea:
    def __init__(self, area):
        self.contour = np.array(area, dtype=np.int32)
        self.polygon = Polygon(area)
        if len(self.contour) <= 2:
            raise Exception('The number of endpoints is less than three!')
        self.count = 0  # 该区域的报警数（离线视频用，实时流不用）
        self.color = (85, 90, 255)
        # 记录该区域，每个报警 id 的报警时间戳（用来计算据距离上次报警的时间）
        self.alarm_time = {}

def read_areas(config):
    areas = []
    try:
        for i,area_points in enumerate(config):
            areas.append(OneArea(area_points))
    except:
        pass
    # return areas, alarm_interval, detect_class
    return areas