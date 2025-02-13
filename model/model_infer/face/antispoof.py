from __future__ import division
import numpy as np
import cv2
import torch
from skimage import transform as trans
from model.model_infer.tools.parser import get_config
from model.model_infer.triton_backend import TritonInfer
import tritonclient.grpc as grpcclient

class AntiSpoofTriron(TritonInfer):

    def infer(self, img1,img2):
        inputs = [grpcclient.InferInput(self.model_input[0], img1.shape, 'FP32'),grpcclient.InferInput(self.model_input[1], img1.shape, 'FP32')]
        inputs[0].set_data_from_numpy(img1)
        inputs[1].set_data_from_numpy(img2)
        # # 创建第一个输入
        # input1 = grpcclient.InferInput(self.model_input[0], [1, 3, 80,80],'FP32')  # 假设是一个图像分类模型，输入尺寸为 224x224
        # input1_data = img1  # 这里是 input1 的数据，需要是一个数组
        # input1.set_data_from_numpy(input1_data)
        #
        # # 创建第二个输入
        # input2 = grpcclient.InferInput(self.model_input[1], [1, 3, 80,80],'FP32')  # 假设是一个语言模型，输入是一个序列长度为 10
        # input2_data = img2  # 这里是 input2 的数据，需要是一个数组
        # input2.set_data_from_numpy(input2_data)
        # inputs=[input1,input2]


        # # 创建 ModelInferRequest 并添加输入
        # request = grpcclient.create_infer_request()
        # request.add_input(input1).add_input(input2)
        # result_infers = request.infer()
        # inputs = [grpcclient.InferInput(self.model_input, img1.shape, 'FP32')]
        # inputs[0].set_data_from_numpy(img1)
        # inputs[1].set_data_from_numpy(img2)
        if isinstance(self.model_output, list):
            result_infers=[]
            # for output_name in self.model_output:
            outputs = [grpcclient.InferRequestedOutput(self.model_output[0]), grpcclient.InferRequestedOutput(self.model_output[1])]
            results = self.triton_client.infer(self.model_name, inputs, request_id=str(111), model_version='', outputs=outputs)
            result_infer1 = results.as_numpy(self.model_output[0])
            result_infer2 = results.as_numpy(self.model_output[1])
            result_infer1 = torch.tensor(result_infer1.astype(np.float32))
            result_infer2 = torch.tensor(result_infer2.astype(np.float32))
            result_infers.append(result_infer1)
            result_infers.append(result_infer2)
        else:
            outputs = [grpcclient.InferRequestedOutput(self.model_output), ]

            results = self.triton_client.infer(self.model_name, inputs, request_id=str(111), model_version='', outputs=outputs)
            result_infers = results.as_numpy(self.model_output)
            result_infers = torch.tensor(result_infers.astype(np.float32))
        return result_infers
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # Subtract the max to stabilize the exponentiation
    return e_x / e_x.sum(axis=1) # Normalize to probabilities
class AntiSpoof:
    def __init__(self,model_name,triton_cfg,conf_thre=0.35, nms_thre=0.45):

        self.triton_sess = AntiSpoofTriron(model_name,triton_cfg)
        self.inputsize = triton_cfg.model_info[model_name].size
        # self.img_height, self.img_width =None,None
        self.stride = 32
        self.confidence_thres = conf_thre
        self.iou_thres = nms_thre
        self.classes = triton_cfg.model_info[model_name].labels
        self.agnostic_nms = False
        self.max_det = 1000
        self.orgshapeList = []
        self.newshapeList = []
    def __call__(self, im0, box=None,conf_threshold=0.4, iou_threshold=0.45, nm=32):
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
        im1,im2= self.preprocess(im0,box)

        # Ort inference
        preds = self.triton_sess.infer(im1,im2)

        # Post-process
        result= self.postprocess(preds)
        return result
    def preprocess(self,img,box_xyxy=None):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Convert the image color space from BGR to RGB
        self.img_height, self.img_width = img.shape[:2]
        x1,y1,x2,y2 = map(int,box_xyxy)
        box_wh=[int(x1),int(y1),int(x2-x1),int(y2-y1)]
        param1 = {
            "org_img": img,
            "bbox": box_wh,
            "scale": 2.7,
            "out_w": 80,
            "out_h": 80,
            "crop": True,
        }
        param2 = {
            "org_img": img,
            "bbox": box_wh,
            "scale": 4.0,
            "out_w": 80,
            "out_h": 80,
            "crop": True,
        }
        img1 = self.crop(**param1)
        img2 = self.crop(**param2)
        input_data1=np.expand_dims(np.transpose(img1, (2, 0, 1)), axis=0)
        input_data1 = input_data1.astype(np.float32)

        input_data2=np.expand_dims(np.transpose(img2, (2, 0, 1)), axis=0)
        input_data2 = input_data2.astype(np.float32)

        return input_data1, input_data2
    def _get_new_box(self,src_w, src_h, bbox,scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min(min((src_h-1)/box_h, (src_w-1)/box_w),scale)

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y

        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1

        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1

        return int(left_top_x), int(left_top_y), \
            int(right_bottom_x), int(right_bottom_y)
    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox,scale)

            img = org_img[left_top_y: right_bottom_y+1,
                  left_top_x: right_bottom_x+1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img
    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        prediction = np.zeros((1, 3))
        # Transpose and squeeze the output to match the expected shape
        output1 = np.transpose(np.squeeze(output[0]))
        output2 = np.transpose(np.squeeze(output[1]))
        outputs=output1+output2#np.ascontiguousarray(outputs)
        # Get the number of rows in the outputs array
        outputs=np.ascontiguousarray(outputs)
        label = np.argmax(outputs)
        return label

if __name__ == '__main__':
    imagepath = '/home/ww/work/project/triton_project/facetest.png'
    img = cv2.imread(imagepath)
    config_path='/home/ww/work/project/triton_project/model/model_infer/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    tritondetector = AntiSpoof('antispoof-ensemble',cfg)
    face_feature = tritondetector(img)
    # img=tritondetector.draw_detections(img,boxes,scores,classes)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    print(face_feature)

