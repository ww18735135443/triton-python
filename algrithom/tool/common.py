import numpy as np
from shapely.geometry import Polygon,LineString,geo

class OneArea:
    def __init__(self, area,index):
        self.region_index=index
        self.contour = np.array(area, dtype=np.int32)
        self.polygon = Polygon(area)
        if len(self.contour) <= 2:
            raise Exception('The number of endpoints is less than three!')
        self.count = 0  # 该区域统计数
        self.color = (85, 90, 255)
        # 记录该区域，每个报警 id 的报警时间戳（用来计算据距离上次报警的时间）
        self.alarm_time = {}

def read_areas(config):
    areas = []
    try:
        for i,area_points in enumerate(config):
            areas.append(OneArea(area_points,i))
    except:
        pass
    # return areas, alarm_interval, detect_class
    return areas

def read_lines(config):
    lines = []
    try:
        for i,line_point in enumerate(config):

            lines.append(One_line_dir(line_point,i))
        for i, line in enumerate(lines):
            indicate_vector = (line.p_d_1[0] - line.p_d_0[0], line.p_d_1[1] - line.p_d_0[1])  # 箭头坐标
            # 没有指定绊线方向
            if indicate_vector[0] == 0 and indicate_vector[1] == 0:
                continue
            # 指定绊线方向
            vertical_vector1 = (line.p0[1] - line.p1[1], line.p1[0] - line.p0[0])
            vertical_vector2 = (line.p1[1] - line.p0[1], line.p0[0] - line.p1[0])
            boundary_vertical_multiply1 = vertical_vector1[0] * indicate_vector[0] + vertical_vector1[1] * \
                                          indicate_vector[1]
            if boundary_vertical_multiply1 >= 0:
                vertical_vector = vertical_vector1
            else:
                vertical_vector = vertical_vector2
            polygon = Polygon([line.p0, line.p1, (line.p1[0] + vertical_vector[0], line.p1[1] + vertical_vector[1])])
            clockwise_state = polygon.exterior.is_ccw
            if clockwise_state:
                line.direction += "CW"
            else:
                line.direction += "CCW"
            line.single_dir = True
            line.ver = vertical_vector
    except:
        pass
    return lines
class One_line_dir:
    def __init__(self, line,line_idx):
        #拌线名称
        self.region_index=line_idx
        # 绊线坐标
        self.p0 = (line[0])
        self.p1 = (line[1])
        # 箭头坐标
        self.p_d_0 = (line[2])
        self.p_d_1 = (line[3])
        # 法向量
        self.ver = None
        # 是否单向绊线
        self.single_dir = False
        # 画线颜色和大小
        self.color = (255, 0, 255)
        self.lineThickness = 2
        self.warn_count = 0  # 该条绊线的报警数（离线视频用，实时流不用）
        # cw or ccw
        self.direction = ""  # 双向绊线

def checkIntersect(t_mid, p_mid, p2, p3):
    """
    功能：计算目标轨迹与画线是否有交点
    输入：
            t_mid是前1s的目标底边中心点坐标
            p_mid是当前帧的目标底部中心点坐标
            p2是画线线段的第一个端点坐标
            p3是画线线段的第二个端点坐标
    输出：
            有交点返回True、没有交点返回False
    """

    line1 = geo.LineString([t_mid, p_mid])
    line2 = geo.LineString([p2, p3])
    cro=line1.intersects(line2)

    return line1.intersects(line2)


def judge_direction(object_prepoint, object_point, line):
    """
    功能：根据计算出的垂直向量，求向量间的夹角，判断该目标与垂直向量是否同向
    输入：
            object_prepoint是前1s的目标底边中心点坐标
            object_point是当前帧的目标底部中心点坐标
    输出：
            目标运动方向与入侵方向相同返回True，与入侵方向相反返回False
    """

    if not line.single_dir:  # 双向绊线
        return True
    vector1 = (object_point[0] - object_prepoint[0], object_point[1] - object_prepoint[1])
    vector_multiply = vector1[0] * line.ver[0] + vector1[1] * line.ver[1]
    if vector_multiply >= 0:
        return True
    else:
        return False

def compute_iou(rec1, rec2):
    """
	计算两个矩形框的交并比。
	:param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
	:param rec2: (x0,y0,x1,y1)
	:return: 交并比IOU.
	"""
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (S1 + S2 - S_cross)


# 面积比
def areaRatio(pre_xyxy, bbox_xyxy):
    S1 = (pre_xyxy[2] - pre_xyxy[0]) * (pre_xyxy[3] - pre_xyxy[1])
    S2 = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
    return S1 / S2


# 上下框变动
def upDownChange(pre_xyxy, bbox_xyxy):
    upCh = bbox_xyxy[1] - pre_xyxy[1]
    downCh = bbox_xyxy[3] - pre_xyxy[3]

    if abs(upCh) == 0:
        return abs(downCh)
    elif abs(downCh) == 0:
        return abs(upCh)
    else:
        return abs(downCh / upCh)


def checkLineCross(boundary_line, pre_xyxy, bbox_xyxy):
    """
    功能：该函数是计算单条线段的报警情况
    输入：
            boundary_lines是当前要计算的画线线段
            pre_xyxy是前1s该目标的左上顶点和右下顶点坐标
            bbox_xyxy是当前帧该目标的坐标
    输出：
            cross是否绊线
    """
    cross = False
    intersect_3 = False
    # 绊线的两个端点
    bLine_p0 = (boundary_line.p0[0], boundary_line.p0[1])
    bLine_p1 = (boundary_line.p1[0], boundary_line.p1[1])

    t_mid = ((pre_xyxy[0] + pre_xyxy[2]) / 2, pre_xyxy[3])  # 前一帧中该 id 的底部中心点
    p_mid = ((bbox_xyxy[0] + bbox_xyxy[2]) / 2, bbox_xyxy[3])  # 当前帧中该 id 的底部中心点
    #p_mid_3 = ((bbox_xyxy[0] + bbox_xyxy[2]) / 2, bbox_xyxy[1] / 3 + 2 * bbox_xyxy[3] / 3)  # 当前帧中该 id 的底部1/3处中心点
    # t_left = (pre_xyxy[0], pre_xyxy[3])  # 前一帧中该 id 的左角点
    # p_left = (bbox_xyxy[0], bbox_xyxy[3])  # 当前帧中该 id 的左角点
    # t_right = (pre_xyxy[2], pre_xyxy[3])  # 前一帧中该 id 的右角点
    # p_right = (bbox_xyxy[2], bbox_xyxy[3])  # 当前帧中该 id 的右角点

    intersect = checkIntersect(t_mid, p_mid, bLine_p0, bLine_p1)
    # intersect_left = checkIntersect(t_left, p_left, bLine_p0, bLine_p1)
    # intersect_right = checkIntersect(t_right, p_right, bLine_p0, bLine_p1)
    # intersect_right_left = checkIntersect(t_right, p_left, bLine_p0, bLine_p1)  # 前一帧右角点与当前帧左角点
    # intersect_left_right = checkIntersect(t_left, p_right, bLine_p0, bLine_p1)  # 前一帧左角点与当前帧右角点

    #if intersect:
    #intersect_3 = checkIntersect(t_mid, p_mid_3, bLine_p0, bLine_p1)

    # 发生绊线的前提下再判断是否和绊线方向一致
    if intersect:
        direction = judge_direction(t_mid, p_mid, boundary_line)
        if direction:
            # if compute_iou(bbox_xyxy, pre_xyxy) < 0.90 and 1 / 8 < upDownChange(pre_xyxy, bbox_xyxy) < 8:
            #     boundary_line.count += 1
            cross = 1

    return cross
def calculate_line_coverage_percentage(line_coords, boxes):
    # 创建线段

    line = LineString([(line_coords.p0[0], line_coords.p0[1]), (line_coords.p1[0], line_coords.p1[1])])

    # 初始化总覆盖长度
    total_covered_length = 0

    # 遍历所有矩形框
    for box_dict in boxes:
        if box_dict["cls"]!="fence":
            continue
        box=box_dict["xyxy"]
        # 创建矩形框（Polygon）
        rect = Polygon([(box[0], box[1]), (box[0], box[3]), (box[2], box[3]), (box[2], box[1])])

        # 计算线段与矩形框的交集
        intersection = line.intersection(rect)

        # 如果交集存在（且不为空），则计算长度
        if intersection and intersection.geom_type == 'LineString':
            total_covered_length += intersection.length

            # 如果交集是多点（意味着线段与矩形框在多个点相交），则计算这些点将线段分成的各个部分
        # 并检查每部分是否完全在矩形内
        elif intersection and intersection.geom_type == 'MultiPoint':
            # 将交点添加到线段的坐标列表中
            intersection_points = list(intersection.coords)
            # 添加线段的起点和终点到列表
            points = [line.coords[0]] + intersection_points + [line.coords[1]]

            # 遍历所有相邻的点对，检查每段是否完全在矩形内
            for i in range(len(points) - 1):
                segment = LineString([points[i], points[i + 1]])
                if rect.contains(segment):
                    total_covered_length += segment.length

                    # 计算线段的总长度
    total_line_length = line.length

    # 计算并返回覆盖百分比
    if total_line_length > 0:
        return (total_covered_length / total_line_length) * 100
    else:
        return 0

class ResultProcess:
    def __init__(self):
        pass
    @staticmethod
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
    @staticmethod
    def mechine_result_process(warn_detect,frame,param,warn_flag,timestamp):
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
        msg["results"]["data"]=warn_detect['detection_object']
        msg["results"]["statitic"]=warn_detect['statics']
        msg["results"]["new_object"]=warn_detect['changs_object']

        # msg={}
        # msg["image"]=base64.b64encode(frame).decode('utf-8')
        msg["image"]=frame
        msg["warnflag"]=warn_flag
        return msg

