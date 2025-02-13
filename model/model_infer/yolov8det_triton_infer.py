import cv2
import numpy as np
from model.model_infer.triton_backend import TritonInfer
import torch
from model.model_infer.tools.common import LetterBox,non_max_suppression,scale_boxes
from model.model_infer.tools.parser import get_config
from model.model_infer.tools.plotting import Colors



class YoloV8TritonDetector:
    def __init__(self,model_name,triton_cfg,conf_thre=0.25, nms_thre=0.7):

        self.triton_sess = TritonInfer(model_name,triton_cfg)
        self.inputsize = triton_cfg.model_info[model_name].size
        # self.img_height, self.img_width =None,None
        self.stride = 32
        self.confidence_thres = conf_thre
        self.iou_thres = nms_thre
        self.classes = triton_cfg.model_info[model_name].labels
        self.agnostic_nms = False
        self.max_det = 1000
        self.color_palette = Colors()
        self.orgshapeList = []
        self.newshapeList = []

    def __call__(self,img):
        input_image = self.preprocess(img)

        outputs = self.triton_sess.infer(input_image)
        boxes,scores,classes=self.postprocess(outputs,input_image,img)
        # result_img=self.draw_detections(img,nms_boxes,nms_scores,nms_classes)
        # cv2.imshow('img',result_img)
        return boxes,scores,classes
    # def preprocess(self,input_image):
    #     """
    #     Preprocesses the input image before performing inference.
    #
    #     Returns:
    #         image_data: Preprocessed image data ready for inference.
    #     """
    #
    #     # Get the height and width of the input image
    #     self.img_height, self.img_width = input_image.shape[:2]
    #
    #     # Convert the image color space from BGR to RGB
    #     img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    #
    #     # Resize the image to match the input shape
    #     img = cv2.resize(img, self.inputsize)
    #
    #     # Normalize the image data by dividing it by 255.0
    #     image_data = np.array(img) / 255.0
    #
    #     # Transpose the image to have the channel dimension as the first dimension
    #     image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
    #
    #     # Expand the dimensions of the image data to match the expected input shape
    #     image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    #
    #     # Return the preprocessed image data
    #     return image_data
    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.inputsize)
        return [letterbox(image=x) for x in [im]]
    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        self.img_height, self.img_width = im.shape[:2]

        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            # cv2.imshow('4', np.array(im[0,...]))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        # im = im.to(self.device)
        # im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im= np.array(im) / 255.0  # 0 - 255 to 0.0 - 1.0
        return im.astype(np.float32)
    """
    后处理
    """
    # def postprocess(self, output):
    #     """
    #     Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.
    #
    #     Args:
    #         input_image (numpy.ndarray): The input image.
    #         output (numpy.ndarray): The output of the model.
    #
    #     Returns:
    #         numpy.ndarray: The input image with detections drawn on it.
    #     """
    #
    #     # Transpose and squeeze the output to match the expected shape
    #     outputs = np.transpose(np.squeeze(output[0]))
    #     outputs=np.ascontiguousarray(outputs)
    #     # Get the number of rows in the outputs array
    #     rows = outputs.shape[0]
    #
    #     # Lists to store the bounding boxes, scores, and class IDs of the detections
    #     boxes = []
    #     scores = []
    #     class_ids = []
    #
    #     # Calculate the scaling factors for the bounding box coordinates
    #     x_factor = self.img_width / self.inputsize[0]
    #     y_factor = self.img_height / self.inputsize[1]
    #
    #     # Iterate over each row in the outputs array
    #     for i in range(rows):
    #         # Extract the class scores from the current row
    #         classes_scores = outputs[i][4:]
    #
    #         # Find the maximum score among the class scores
    #         max_score = np.amax(classes_scores)
    #
    #         # If the maximum score is above the confidence threshold
    #         if max_score >= self.confidence_thres:
    #             # Get the class ID with the highest score
    #             class_id = np.argmax(classes_scores)
    #
    #             # Extract the bounding box coordinates from the current row
    #             x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
    #
    #             # Calculate the scaled coordinates of the bounding box
    #             left = int((x - w / 2) * x_factor)
    #             top = int((y - h / 2) * y_factor)
    #             width = int(w * x_factor)
    #             height = int(h * y_factor)
    #
    #             # Add the class ID, score, and box coordinates to the respective lists
    #             class_ids.append(class_id)
    #             scores.append(max_score)
    #             boxes.append([left, top, width, height])
    #
    #     # Apply non-maximum suppression to filter out overlapping bounding boxes
    #     indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
    #
    #     # # Iterate over the selected indices after non-maximum suppression
    #     # for i in indices:
    #     #     # Get the box, score, and class ID corresponding to the index
    #     #     box = boxes[i]
    #     #     score = scores[i]
    #     #     class_id = class_ids[i]
    #
    #         # Draw the detection on the input image
    #         # self.draw_detections(input_image, box, score, class_id)
    #     nms_boxes=[boxes[i] for i in indices]
    #     nms_scores=[scores[i] for i in indices]
    #     nms_classes=[class_ids[i] for i in indices]
    #     # Return the modified input image
    #     # return nms_boxes,nms_scores,nms_classes
    #
    #     # nms_boxes=scale_boxes(self.inputsize, nms_boxes, [self.img_width,self.img_height])
    #     # nms_boxes=nms_boxes.astype(np.int32)
    #     # nms_boxes=nms_boxes.tolist()
    #
    #     return nms_boxes,nms_scores,nms_classes

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = non_max_suppression(
            preds,
            self.confidence_thres,
            self.iou_thres
        )


        pred=preds[0]
        boxes= scale_boxes(img.shape[2:], pred[:, :4], orig_imgs.shape)
        boxes=boxes.astype(np.int32).tolist()
        scores=np.array(pred[:, -2]).tolist()
        classes=np.array(pred[:, -1]).astype(np.int8).tolist()
        return boxes,scores,classes
    def draw_detections(self, img, bboxs, scores, class_ids):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        for box, score, class_id in zip(bboxs, scores, class_ids):
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = box

            # Retrieve the color for the class ID
            color = self.color_palette.palette[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Create the label text with class name and score
            label = f'{self.classes[class_id]}: {score:.2f}'

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                          cv2.FILLED)

            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img


def draw_detections(self, img, bboxs, scores, class_ids):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        for box, score, class_id in zip(bboxs, scores, class_ids):
            # Extract the coordinates of the bounding box
            x1, y1, w, h = box

            # Retrieve the color for the class ID
            color = self.color_palette.palette[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            # Create the label text with class name and score
            label = f'{self.classes[class_id]}: {score:.2f}'

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                          cv2.FILLED)

            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img







    # def postprocess(self, prediction):
    #     prediction = torch.tensor(prediction)
    #     preds = non_max_suppression(prediction, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
    #                                 max_det=self.max_det)
    #     for i, pred in enumerate(preds):
    #         pred[:, :4] = scale_coords(self.newshapeList[i], pred[:, :4], self.orgshapeList[i]).round()
    #     self.orgshapeList = []
    #     self.newshapeList = []
    #     return preds
    #
    # """
    # 一张图推理
    # """
    # def __call__(self, image):
    #     imgbatch_tensor = self.preprocess([image])
    #     pred_onx = self.triton_sess.infer(imgbatch_tensor)
    #     pred_onx = pred_onx[0]
    #     preds = self.postprocess(pred_onx)[0]
    #     bboxes, cls, scores = preds[:, :4], preds[:, 5], preds[:, 4]
    #     return bboxes, cls, scores

if __name__ == '__main__':
    imagepath = '/home/ww/work/project/triton_project/3.jpg'
    img = cv2.imread(imagepath)
    config_path='/home/ww/work/project/triton_project/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    tritondetector = YoloV8TritonDetector('fencemodel',cfg)
    boxes,scores,classes = tritondetector(img)
    print(boxes)
    img=tritondetector.draw_detections(img,boxes,scores,classes)

    cv2.imshow('4', img)
    cv2.waitKey(0)