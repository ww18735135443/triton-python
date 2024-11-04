import tritonclient.grpc as grpcclient
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from model.model_infer.tools.parser import get_config

class TritonInfer:
    def __init__(self,model_name,cfg='model/model_infer/config/triton_config.yaml' ,Norm=0):
        self.model_name = model_name
        if isinstance(cfg,str):
            cfg = get_config()
            cfg.merge_from_file(config_path)
        self.triton_server_url = cfg.url
        self.model_input=cfg.model_info[self.model_name].input
        self.model_output=cfg.model_info[self.model_name].output
        self.size = tuple(cfg.model_info[self.model_name].size)
        self.triton_client = grpcclient.InferenceServerClient(url=self.triton_server_url, verbose=False)
        if Norm:
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    def infer(self, img):
        inputs = [grpcclient.InferInput(self.model_input, img.shape, 'FP32')]
        inputs[0].set_data_from_numpy(img)
        if isinstance(self.model_output, list):
            result_infers=[]
            for output_name in self.model_output:
                outputs = [grpcclient.InferRequestedOutput(output_name), ]
                results = self.triton_client.infer(self.model_name, inputs, request_id=str(111), model_version='', outputs=outputs)
                result_infer = results.as_numpy(output_name)
                result_infer = torch.tensor(result_infer.astype(np.float32))
                result_infers.append(result_infer)
        else:
            outputs = [grpcclient.InferRequestedOutput(self.model_output), ]

            results = self.triton_client.infer(self.model_name, inputs, request_id=str(111), model_version='', outputs=outputs)
            result_infers = results.as_numpy(self.model_output)
            result_infers = torch.tensor(result_infers.astype(np.float32))
        return result_infers
# def get_config(config_file=None):
#     return YamlParser(config_file=config_file)

if __name__ == '__main__':
    config_path='/home/ww/work/project/triton_project/config/triton_config.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_path)
    extractor = TritonInfer('scsmodel')
    imagepath = '/home/ww/work/project/triton_project/157368844_23.jpg'
    import cv2
    img = cv2.imread(imagepath)
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # h*w*c convert to c*h*w
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    result = extractor.infer(img)
    print(result)
