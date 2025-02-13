import tritonclient.grpc as grpcclient
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms


class TritonInfer:
    def __init__(self, triton_server_url,cfg = None):
        self.model_name = 'scsmodel'
        if cfg:
            triton_server_url = cfg.tritoninfer.url
            self.model_name = cfg.tritoninfer.model_name
            self.model_input=cfg.tritoninfer.model_name.input
            self.model_output=cfg.tritoninfer.model_name.output
            self.size = tuple(cfg.tritoninfer.model_name.size)
        self.triton_client = grpcclient.InferenceServerClient(url=triton_server_url, verbose=False)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    def infer(self, img):
        outputs = [grpcclient.InferRequestedOutput(self.model_output), ]
        inputs = [grpcclient.InferInput(self.model_input, img.shape, 'FP32')]
        inputs[0].set_data_from_numpy(img)
        results = self.triton_client.infer(self.model_name, inputs, request_id=str(111), model_version='', outputs=outputs)
        result_infer = results.as_numpy(self.model_output)
        result_infer = torch.tensor(result_infer.astype(np.float32))
        return result_infer

if __name__ == '__main__':
        extractor = TritonInfer('10.5.68.13:8001')
        imagepath = '/mnt/zj/datasets/prepare_dataset/157368844_23.jpg'
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
