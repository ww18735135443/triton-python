from __future__ import division
import numpy as np
import cv2
import onnxruntime
from skimage import transform as trans
from model.model_infer.tools.parser import get_config
from model.model_infer.triton_backend import TritonInfer
import onnxruntime as ort
arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

class ArcFaceONNX:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = 'recognition'
        find_sub = False
        find_mul = False
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        #print('input mean and std:', self.input_mean, self.input_std)
        if self.session is None:
            self.session = ort.InferenceSession(self.model_file, providers=['CUDAExecutionProvider'])
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names)==1
        self.output_shape = outputs[0].shape

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img, face):
        aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

class ArcFaceTriton:
    def __init__(self, model_name,triton_cfg,conf_thre=0.35, nms_thre=0.45):
        if isinstance(triton_cfg,str):
            triton_cfg_ = get_config()
            triton_cfg_.merge_from_file(triton_cfg)
            triton_cfg= triton_cfg_
        self.triton_cfg = triton_cfg
        self.input_size=triton_cfg.model_info[model_name].size
        self.triton_sess = TritonInfer(model_name,triton_cfg)
        self.inputsize = triton_cfg.model_info[model_name].size
        # self.img_height, self.img_width =None,None
        self.stride = 32
        self.confidence_thres = conf_thre
        self.iou_thres = nms_thre
        self.agnostic_nms = False
        self.max_det = 1000
        self.orgshapeList = []
        self.newshapeList = []
        self.input_mean = 127.5
        self.input_std = 127.5


    def get(self, img, kss):
        aimg = norm_crop(img, landmark=kss, image_size=self.input_size[0])
        embedding = self.get_feat(aimg).flatten()
        return embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        # net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        net_out = self.triton_sess.infer(blob)
        return net_out[0]


if __name__ == '__main__':
    imagepath = '/home/ww/work/project/triton_project/facetest.png'
    img = cv2.imread(imagepath)
    config_path='/home/ww/work/project/triton_project/model/model_infer/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    tritondetector = ArcFaceTriton('facerecognitionmodel',cfg)
    face_feature = tritondetector.get_feat(img)
    # img=tritondetector.draw_detections(img,boxes,scores,classes)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    print(face_feature)