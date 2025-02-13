import cv2
import numpy as np
from model.model_infer.triton_backend import TritonInfer
import torch
from model.model_infer.tools.common import letterbox,non_max_suppression,scale_coords
from model.model_infer.tools.parser import get_config
import torch.nn.functional as F
from model.model_infer.tools.plotting import Colors
class YoloV8segTritonDetector:
    def __init__(self,model_name,triton_cfg,conf_thre=0.5, nms_thre=0.45):

        self.triton_sess = TritonInfer(model_name,triton_cfg)
        self.inputsize = triton_cfg.model_info[model_name].size

        self.stride = 32
        self.confidence_thres = conf_thre
        self.iou_thres = nms_thre
        self.classes = triton_cfg.model_info[model_name].labels
        self.agnostic_nms = False
        self.max_det = 1000
        self.color_palette = Colors()
        self.orgshapeList = []
        self.newshapeList = []
    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """

        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.triton_sess.infer(im)

        # Post-process
        boxes, segments, masks = self.postprocess(preds,
                                                  im0=im0,
                                                  ratio=ratio,
                                                  pad_w=pad_w,
                                                  pad_h=pad_h,
                                                  conf_threshold=conf_threshold,
                                                  iou_threshold=iou_threshold,
                                                  nm=nm)
        return boxes, segments, masks

    """
    批量预处理
    """
    def preprocess_(self, img_list):
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
    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """

        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = tuple(self.inputsize)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum('HWC->CHW', img.astype(np.float32))[::-1]) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    @staticmethod
    def masks2segments(masks):
        """
        It takes a list of masks(n,h,w) and returns a list of segments(n,xy) (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L750)

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype('uint8'):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype('float32'))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        # masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks=torch.tensor(masks_in.astype('float32'))
        masks=(masks @ protos.float().view(c,-1)).sigmoid().view(-1,mh,mw)
        # masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.ascontiguousarray(masks)

        # masks = np.einsum('HWN -> NHW', masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[-2:]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[:,top:bottom, left:right]
        # masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]),
        #                    interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC would be better
        masks=F.interpolate(masks[None], size=im0_shape[:2], mode='bilinear', align_corners=True)[0]
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    # def draw_and_visualize(self, im, bboxes, segments, vis=False, save=True):
    #     """
    #     Draw and visualize results.
    #
    #     Args:
    #         im (np.ndarray): original image, shape [h, w, c].
    #         bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
    #         segments (List): list of segment masks.
    #         vis (bool): imshow using OpenCV.
    #         save (bool): save image annotated.
    #
    #     Returns:
    #         None
    #     """
    #
    #     # Draw rectangles and polygons
    #     im_canvas = im.copy()
    #     for (*box, conf, cls_), segment in zip(bboxes, segments):
    #         # draw contour and fill mask
    #         cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
    #         cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))
    #
    #         # draw bbox rectangle
    #         cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
    #                       self.color_palette(int(cls_), bgr=True), 1, cv2.LINE_AA)
    #         cv2.putText(im, f'{self.classes[cls_]}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_palette(int(cls_), bgr=True), 2, cv2.LINE_AA)
    #
    #     # Mix image
    #     # im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
    #
    #     # # Show image
    #     # if vis:
    #     #     cv2.imshow('demo', im)
    #     #     cv2.waitKey(0)
    #     #     cv2.destroyAllWindows()
    #     #
    #     # # Save image
    #     # if save:
    #     #     cv2.imwrite('demo.jpg', im)
    #     return im

    """
    后处理
    """
    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum('bcn->bnc', x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # Decode and return
        if len(x) > 0:

            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []
    def draw_detections(self, im, bboxes, segments, vis=False, save=True):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """

        # Draw rectangles and polygons
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # draw contour and fill mask
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
            cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))

            # draw bbox rectangle
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          self.color_palette(int(cls_), bgr=True), 1, cv2.LINE_AA)
            cv2.putText(im, f'{self.classes[int(cls_)]}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_palette(int(cls_), bgr=True), 2, cv2.LINE_AA)

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # # Show image
        # if vis:
        #     cv2.imshow('demo', im)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #
        # # Save image
        # if save:
        #     cv2.imwrite('demo.jpg', im)
        return im

if __name__ == '__main__':
    imagepath = '/home/ww/work/project/triton_project/157368844_23.jpg'
    img = cv2.imread(imagepath)
    config_path='/home/ww/work/project/triton_project/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    tritondetector = YoloV8TritonDetector('scsmodel',cfg)
    import time
    for i in range(10):
        start_time=time.time()
        boxes, segments, masks = tritondetector(img)
        end_time=time.time()
        print(end_time-start_time)
    # print(boxes, segments, masks)
    if len(boxes) > 0:
        tritondetector.draw_detections(img, boxes, segments, vis=False, save=True)
    # bboxes = bboxes.cpu()
    # for bbox in bboxes:
    #     bbox = bbox.numpy()
    #     if bbox[0] < 0:
    #         bbox[0] = 0
    #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0.667, 0.000, 1.000, ],
    #                   2)
    # cv2.imshow('test', cv2.resize(img, (1280, 720)))
    # cv2.waitKey(0)
    # print(len(bboxes))


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
