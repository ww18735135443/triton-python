import cv2
import numpy as np
from triton_backend import TritonInfer
import torch
from algrithom.model_infer.tools.common import letterbox,non_max_suppression,scale_coords

class TritonDetector:
    def __init__(self,triton_url,triton_cfg=None, inputsize=(640, 640), conf_thre=0.5, nms_thre=0.45):
        self.triton_sess = TritonInfer(triton_url,triton_cfg)
        self.inputsize = inputsize
        self.stride = 32
        self.conf_thres = conf_thre
        self.iou_thres = nms_thre
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.orgshapeList = []
        self.newshapeList = []

    """
    批量预处理
    """
    def preprocess(self, img_list):
        letterboxed_list = []
        for img in img_list:
            self.orgshapeList.append(img.shape)
            letterboxed_img = letterbox(img.copy(), self.inputsize, stride=self.stride, auto=False)[0]
            self.newshapeList.append(letterboxed_img.shape)
            letterboxed_list.append(letterboxed_img)
        imgbatch = np.stack(letterboxed_list, 0)
        imgbatch = imgbatch[..., ::-1].transpose((0, 3, 1, 2)).astype(np.float32)  # BGR to RGB, BHWC to BCHW
        imgbatch /= 255  # 0 - 255 to 0.0 - 1.0
        return imgbatch

    """
    后处理
    """
    def postprocess(self, prediction):
        prediction = torch.tensor(prediction)
        preds = non_max_suppression(prediction, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                    max_det=self.max_det)
        for i, pred in enumerate(preds):
            pred[:, :4] = scale_coords(self.newshapeList[i], pred[:, :4], self.orgshapeList[i]).round()
        self.orgshapeList = []
        self.newshapeList = []
        return preds

    """
    一张图推理
    """
    def __call__(self, image):
        imgbatch_tensor = self.preprocess([image])
        pred_onx = self.triton_sess.infer(imgbatch_tensor)
        pred_onx = pred_onx[0]
        preds = self.postprocess(pred_onx)[0]
        bboxes, cls, scores = preds[:, :4], preds[:, 5], preds[:, 4]
        return bboxes, cls, scores

if __name__ == '__main__':

    tritondetector = TritonDetector('10.5.68.13:8001')
    img = cv2.imread('/mnt/zj/datasets/prepare_dataset/157368844_23.jpg')
    bboxes, cls, scores = tritondetector(img)
    bboxes = bboxes.cpu()
    for bbox in bboxes:
        bbox = bbox.numpy()
        if bbox[0] < 0:
            bbox[0] = 0
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0.667, 0.000, 1.000, ],
                      2)
    cv2.imshow('test', cv2.resize(img, (1280, 720)))
    cv2.waitKey(0)
    print(len(bboxes))


    """
    多张图推理
    """
    # def batchInfer(self, imageList):
    #     imgbatch_tensor = self.preprocess(imageList)
    #     with torch.no_grad():
    #         prediction = self.model(imgbatch_tensor, augment=False, visualize=False)
    #     preds = self.postprocess(prediction)
    #     results = []
    #     for pred in preds:
    #         results.append((pred[:, :4], pred[:, 5], pred[:, 4]))
    #     return results
