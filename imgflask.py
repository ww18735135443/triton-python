import base64
from model.model_infer.tools.parser import get_config,load_config_from_file
import cv2
from model.model_infer.yolov8det_triton_infer import YoloV8TritonDetector
from model.model_infer.yolov8seg_triton_infer  import YoloV8segTritonDetector
from model.model_infer.yolov5det_triton_infer  import YoloV5TritonDetector
from model.model_infer.face_triton_infer import FaceAlgrithom
from flask import Flask, request, jsonify
from algrithom.tool.draw import draw_areas,draw_detections,draw_detections_boxonly
import numpy as np
app = Flask(__name__)


class DetectRun:
    def __init__(self):
        triton_config_path='model/model_infer/config/triton_config.yaml'
        triton_config = get_config()
        triton_config.merge_from_file(triton_config_path)
        self.steeldetector =YoloV8TritonDetector('steelcount',triton_config)
        self.weardetect=YoloV8TritonDetector('wearmodel',triton_config)
        self.scsdetect=YoloV8segTritonDetector('scsmodel',triton_config)
        self.fencedetect=YoloV5TritonDetector('fencemodel',triton_config)
        self.smokefiredetect=YoloV5TritonDetector('smokefiremodel',triton_config)
        self.facemodel=FaceAlgrithom(triton_config)
        self.triton_config=triton_config
    def warn_up(self):
        random_img=np.random.randint(0,255,(640,640,3),dtype=np.uint8)
        self.steeldetector(random_img)
        self.weardetect(random_img)
        self.scsdetect(random_img)
        self.fencedetect(random_img)
        self.smokefiredetect(random_img)

def base64_2_img(img_base64):
    img_decoded=base64.b64decode(img_base64)
    img_array=np.frombuffer(img_decoded,dtype=np.uint8)
    img=cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    return img
def detect_data_process(boxes,scores,cls,labels=None):
    detections=[]
    for box, score, class_id in zip(boxes,scores,cls):
        detection={}
        detection['xyxy']=box
        if labels is not None:
            detection['cls']=labels[int(class_id)]
        else:
            detection['cls']=int(class_id)
        detection['class_id']=int(class_id)
        detection['conf']=float(score)
        detections.append(detection)
    return detections
def seg_data_process(detections, segments, masks, labels=None):
    new_detections = []
    if len(detections)==0:
        return new_detections
    else:
        for i in range(len(detections)):
            detection={}
            detection['xyxy']=np.array(detections[i,:4]).tolist()
            if labels is not None:
                detection['cls']=labels[int(np.array(detections[i,5]))]
            else:
                detection['cls']=np.array(detections[i,5])
            detection['class_id']=int(np.array(detections[i,5]))
            detection['conf']=float(np.array(detections[i,4]))
            new_detections.append(detection)
    return new_detections
def results_filter(detections,labels):
    detections_filtered=[]
    for detection in detections:
        if detection['cls'] in labels:
            detections_filtered.append(detection)
    return detections_filtered
def result_process(frame,detect_result):
    success, buffer = cv2.imencode('.jpg', frame)
    # if not success:
    #     raise RuntimeError("无法将图像编码为PNG格式")

    # 将字节流转换为字节串（通常这一步是自动的，但为了确保类型正确，可以显式转换）
    img_bytes = buffer.tobytes()  # 或者直接使用 buffer，因为它已经是一个可迭代的字节对象

    # 将字节串编码为 Base64 字符串
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    # with open('test20241216.txt', 'w') as text_file:
    #     text_file.write(img_base64)
    # im=base64_2_img(img_base64)
    # cv2.imshow('steelcount',im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    msg={}
    msg["image"]=img_base64 #base64.b64encode(frame).decode('utf-8')
    msg["detects"]=detect_result
    return msg
def result_process_with_num(frame,detect_result):
    success, buffer = cv2.imencode('.jpg', frame)
    # 将字节流转换为字节串（通常这一步是自动的，但为了确保类型正确，可以显式转换）
    img_bytes = buffer.tobytes()  # 或者直接使用 buffer，因为它已经是一个可迭代的字节对象

    # 将字节串编码为 Base64 字符串
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    msg={}
    msg["image"]=img_base64 #base64.b64encode(frame).decode('utf-8')
    msg["detects"]=detect_result
    msg["num"]=len(detect_result)
    return msg

detector = DetectRun()
detector.warn_up()

@app.route('/facecompare', methods=['POST'])
def facecompare():
    data = request.json
    try:
        #获取输入并转为图片
        if "major_img" in data.keys():
            major_img=base64_2_img(data["major_img"])
            other_imgs_base64=data["other_img"]
            other_imgs=[]
            for index,other_img in other_imgs_base64.items():
                other_img_dict={}
                other_img_dict[index]=base64_2_img(other_img)
                other_imgs.append(other_img_dict)
        #获取主要图片人脸特征
        major_fea=detector.facemodel.featureget(major_img,1)
        #获取配图人脸特征相似度
        sim_results=[]
        for img_dict in other_imgs:
            sim_result={}
            for img_idx,img_fig in img_dict.items():
                idx_fea=detector.facemodel.featureget(img_fig,1)
                face,kss= detector.facemodel.facedetect(img_fig)
                sim_score=detector.facemodel.feature_compare(major_fea,idx_fea)
                sim_result[img_idx]=sim_score
                if len(face)>0:
                    sim_result['box']=face[0][:4].tolist()
                else:
                    sim_result['box']=[]
                sim_results.append(sim_result)
        return jsonify({"success":sim_results})
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)})

@app.route('/steelcount', methods=['POST'])
def steelcount():
    data = request.json
    try:
        if "image" in data.keys():
            img_base64=data["image"]
            img=base64_2_img(img_base64)
            boxes,scores,cls = detector.steeldetector(img)
            detect_info=detector.triton_config
            model_label=detect_info.model_info['steelcount']['labels']
            detections=detect_data_process(boxes,scores,cls,model_label)
            draw_detections_boxonly(img,detections,1)
            # cv2.imshow('steelcount',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            msg=result_process_with_num(img,detections)
            return jsonify({"success":msg})
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)})

@app.route('/smokefiredetect', methods=['POST'])
def smokefiredetect():
    data = request.json
    try:
        if "image" in data.keys():
            img_base64=data["image"]
            img=base64_2_img(img_base64)
            boxes,scores,cls = detector.smokefiredetect(img)
            detect_info=detector.triton_config
            model_label=detect_info.model_info['smokefiremodel']['labels']
            detections=detect_data_process(boxes,scores,cls,model_label)
            draw_detections(img,detections,1)
            msg=result_process(img,detections)
            return jsonify({"success":msg})
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)})

@app.route('/fencedetect', methods=['POST'])
def fencedetect():
    data = request.json
    try:
        if "image" in data.keys():
            img_base64=data["image"]
            img=base64_2_img(img_base64)
            boxes,scores,cls = detector.fencedetect(img)
            detect_info=detector.triton_config
            model_label=detect_info.model_info['fencemodel']['labels']
            detections=detect_data_process(boxes,scores,cls,model_label)
            draw_detections(img,detections,1)
            msg=result_process(img,detections)
            return jsonify({"success":msg})
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)})

@app.route('/helmetdetect', methods=['POST'])
def helmetdetect():
    data = request.json
    try:
        if "image" in data.keys():
            img_base64=data["image"]
            img=base64_2_img(img_base64)
            boxes,scores,cls = detector.weardetect(img)
            detect_info=detector.triton_config
            model_label=detect_info.model_info['wearmodel']['labels']
            detections=detect_data_process(boxes,scores,cls,model_label)
            # object_labels=['helmet','nohelmet','person']
            object_labels=['helmet','nohelmet']
            detections=results_filter(detections,object_labels)
            draw_detections(img,detections,1)

            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            msg=result_process(img,detections)
            return jsonify({"success":msg})
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)})

@app.route('/safetybeltdetect', methods=['POST'])
def safetybeltdetect():
    data = request.json
    try:
        if "image" in data.keys():
            img_base64=data["image"]
            img=base64_2_img(img_base64)
            boxes,scores,cls = detector.weardetect(img)
            detect_info=detector.triton_config
            model_label=detect_info.model_info['wearmodel']['labels']
            detections=detect_data_process(boxes,scores,cls,model_label)
            object_labels=['belt']
            detections=results_filter(detections,object_labels)
            draw_detections(img,detections,1)
            msg=result_process(img,detections)
            return jsonify({"success":msg})
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)})

@app.route('/reflectivedetect', methods=['POST'])
def reflectivedetect():
    data = request.json
    try:
        if "image" in data.keys():
            img_base64=data["image"]
            img=base64_2_img(img_base64)
            boxes,scores,cls = detector.weardetect(img)
            detect_info=detector.triton_config
            model_label=detect_info.model_info['wearmodel']['labels']
            detections=detect_data_process(boxes,scores,cls,model_label)
            object_labels=['reflectivevest']
            detections=results_filter(detections,object_labels)
            draw_detections(img,detections,1)
            # cv2.imshow('steelcount',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            msg=result_process(img,detections)
            return jsonify({"success":msg})
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)})

@app.route('/vasdetect', methods=['POST'])
def vasdetect():
    data = request.json
    try:
        if "image" in data.keys():
            img_base64=data["image"]
            img=base64_2_img(img_base64)
            detections, segments, masks  = detector.scsdetect(img)
            detect_info=detector.triton_config
            model_label=detect_info.model_info['scsmodel']['labels']
            detections=seg_data_process(detections, segments, masks,model_label)
            object_labels=["person","worker","excavator","loader","dumptruck","truckcrane","crawlercrane",
                           "concretemixertruck","pumptruck","trailerpump","passengervehicle","rider","boxtruck","towercrane",
                           "roller","elevator","pctruck","mixer","lighttruck","dozer","forklift"]
            detections=results_filter(detections,object_labels)
            draw_detections(img,detections,1)
            msg=result_process(img,detections)
            return jsonify({"success":msg})
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)})

@app.route('/persondetect', methods=['POST'])
def persondetect():
    data = request.json
    try:
        if "image" in data.keys():
            img_base64=data["image"]
            img=base64_2_img(img_base64)
            detections, segments, masks  = detector.scsdetect(img)
            detect_info=detector.triton_config
            model_label=detect_info.model_info['scsmodel']['labels']
            detections=seg_data_process(detections, segments, masks,model_label)
            object_labels=["person","worker","rider"]
            detections=results_filter(detections,object_labels)
            draw_detections(img,detections,1)
            msg=result_process_with_num(img,detections)
            return jsonify({"success":msg})
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)})


@app.route('/mechinedetect', methods=['POST'])
def mechinedetect():
    data = request.json
    try:
        if "image" in data.keys():
            img_base64=data["image"]
            img=base64_2_img(img_base64)
            detections, segments, masks  = detector.scsdetect(img)
            detect_info=detector.triton_config
            model_label=detect_info.model_info['scsmodel']['labels']
            detections=seg_data_process(detections, segments, masks,model_label)
            object_labels=["excavator","loader","dumptruck","truckcrane","crawlercrane",
                           "concretemixertruck","pumptruck","trailerpump","passengervehicle","boxtruck","towercrane",
                           "roller","elevator","pctruck","mixer","lighttruck","dozer","forklift"]
            detections=results_filter(detections,object_labels)
            draw_detections(img,detections,1)
            # cv2.imshow('mechine',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            msg=result_process(img,detections)
            return jsonify({"success":msg})
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)})
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=9008,debug=True)