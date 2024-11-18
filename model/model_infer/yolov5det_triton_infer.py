import cv2
import numpy as np
from model.model_infer.triton_backend import TritonInfer
import torch
from model.model_infer.tools.common import letterbox,non_max_suppression,scale_coords
from model.model_infer.tools.parser import get_config
from model.model_infer.tools.plotting import Colors
import imageio
class YoloV5TritonDetector:
    def __init__(self,model_name,triton_cfg,conf_thre=0.35, nms_thre=0.45):

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
        im= self.preprocess(im0)

        # Ort inference
        preds = self.triton_sess.infer(im)

        # Post-process
        boxes,scores,classes = self.postprocess(preds)
        return boxes,scores,classes
    def preprocess(self,img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Convert the image color space from BGR to RGB
        self.img_height, self.img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, self.inputsize)

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data
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
    #     class_ids = [class_ids[i] for i in indices]
    #     scores = [scores[i] for i in indices]
    #     boxes = [boxes[i] for i in indices]
    #     return boxes,scores,class_ids
    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))
        outputs=np.ascontiguousarray(outputs)
        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.inputsize[0]
        y_factor = self.img_height / self.inputsize[1]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, left+width, top+height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # # Iterate over the selected indices after non-maximum suppression
        # for i in indices:
        #     # Get the box, score, and class ID corresponding to the index
        #     box = boxes[i]
        #     score = scores[i]
        #     class_id = class_ids[i]
        #
        #     # Draw the detection on the input image
        #     self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        class_ids = [class_ids[i] for i in indices]
        scores = [scores[i] for i in indices]
        boxes = [boxes[i] for i in indices]
        return boxes,scores,class_ids
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
            # label = f'{self.classes[class_id]}'
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



def run(video_path):
    config_path='/home/ww/work/project/triton_project/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    tritondetector = YoloV5TritonDetector('safetymodel',cfg)


    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    videowriter = imageio.get_writer('/home/ww/work/project/triton_project/data/result_videos/helmet.mp4', fps=fps)
    while cap.isOpened():
        # 从视频读取一帧
        rval, ori_img = cap.read()
        if rval:
            boxes,scores,classes = tritondetector(ori_img)
            result_img=tritondetector.draw_detections(ori_img,boxes,scores,classes)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            videowriter.append_data(result_img)



if __name__ == '__main__':
    imagepath = '/home/ww/work/project/triton_project/fence.jpg'
    img = cv2.imread(imagepath)
    config_path='/home/ww/work/project/triton_project/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    tritondetector = YoloV5TritonDetector('fencemodel',cfg)
    boxes,scores,classes = tritondetector(img)
    img=tritondetector.draw_detections(img,boxes,scores,classes)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    # video_path='/home/ww/work/project/triton_project/data/org_videos/helmet.mp4'
    # run(video_path)


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

