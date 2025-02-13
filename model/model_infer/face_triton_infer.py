import cv2
import numpy as np
from model.model_infer.triton_backend import TritonInfer
from model.model_infer.face.facedetect import FaceDetect
from model.model_infer.face.antispoof import AntiSpoof
from model.model_infer.face.arcface import ArcFaceTriton

def normalize(feature1):
    """
    对二维数组feature1进行L2归一化。

    参数:
    feature1 (numpy.ndarray): 二维数组，形状为 (n_samples, n_features)

    返回:
    numpy.ndarray: 归一化后的数组，形状与输入相同
    """
    norm = np.linalg.norm(feature1, axis=1, keepdims=True)
    normalized_feature1 = feature1 / norm
    return normalized_feature1
def find_largest_bbox(faces,kss):
    """
    从检测结果列表中找出面积最大的边界框。

    参数:
    detect_results (list of dict): 检测结果列表，每个元素是一个包含'bbox'键的字典。

    返回:
    tuple: 面积最大的边界框（作为列表）和对应的面积。
    """
    # 初始化最大面积和对应的边界框
    max_area = 0
    max_detect=None
    max_face_kss=None
    # 遍历检测结果
    for detect,kss in zip(faces,kss):
        bbox = detect[:4]
        x1, y1, x2, y2 = bbox

        # 计算边界框的面积
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # 更新最大面积和对应的边界框
        if area > max_area:
            max_area = area
            max_detect=detect
            max_face_kss=kss

    # 返回最大面积的边界框和面积
    return [max_detect],[max_face_kss]
class FaceAlgrithom(TritonInfer):
    def __init__(self,triton_cfg):
        self.facedetector=FaceDetect('facedetectmodel',triton_cfg)
        self.antispoof=AntiSpoof('antispoof-ensemble',triton_cfg)
        self.arcface=ArcFaceTriton('facerecognitionmodel',triton_cfg)
    def facedetect(self,img,select_max=0):
        faces,kss=self.facedetector(img)
        if select_max and len(faces) > select_max:
            faces,kss=find_largest_bbox(faces,kss)
        return faces,kss

    def antispoof(self,face,box):
        spoofresult=self.antispoof(face,box)
        return spoofresult

    def featurextract(self,img,kss):
        face_fea=self.arcface.get(img,kss)
        return face_fea

    def featureget(self,im,select_max=0):
        face,kss=self.facedetect(im,select_max)
        if len(face)==0:
            return None
        if select_max:
            features=self.featurextract(im,kss[0])
        else:
            features=[]
            for face,kss in zip(face,kss):
                face_fea=self.featurextract(im,kss)
                features.append(face_fea)
        return features
    def feature_compare(self,feature1, feature2):
        if feature1 is None or len(feature1)== 0 or feature2 is None or len(feature2)== 0:
            return 0.0
        feature1 = np.array(feature1).reshape((1, -1))
        feature1 = normalize(feature1)
        feature2 = np.array(feature2).reshape((1, -1))
        feature2 = normalize(feature2)
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        d=dist.tolist()[0]
        if d>=2.48:
            return 0.0
        else:
            return (2.48-d)/2.48
        return dist.tolist()[0]


if __name__ == '__main__':
    from model.model_infer.tools.parser import get_config
    imagepath = '/home/ww/work/project/triton_project/face2.png'
    img = cv2.imread(imagepath)
    config_path='/home/ww/work/project/triton_project/model/model_infer/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    tritondetector = FaceAlgrithom(cfg)
    face,kss= tritondetector.facedetect(img)
    print(face)
    antispoof = tritondetector.antispoof(img, face[0][:4])
    print(antispoof)
    face_fea=tritondetector.featurextract(img,kss[0])
    print(len(face_fea))
    fe=tritondetector.featureget(img,)
    print(len(fe))