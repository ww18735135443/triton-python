
import time
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import queue
from functools import partial
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue(maxsize = 10)
def completion_callback(user_data, result, error):
    user_data._completed_requests.put((result, error))
if __name__ == '__main__':
    #预处理
    imgpath = '/mnt/zj/datasets/prepare_dataset/steel_test/c14fa2a403b5f87ceb934a1de43f2ea2.jpg'
    img0 = cv2.imread(imgpath)
    img = cv2.resize(img0,(640,640))
    img = img[:, :, ::-1].transpose(2, 0, 1) # h*w*c convert to c*h*w
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32)
    img /= 255.0 # 0 - 255 to 0.0 - 1.0
    img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
    #初始化用户数据对象，只需在算法启动时初始化1次，该对象的成员函数仅包含一个队列，用于triton进行模型推理后传回结果
    user_data = UserData()
    #初始化triton_client对象，算法启动时初始化一次，然后调用该对象就行了
    triton_client = grpcclient.InferenceServerClient(
    url='10.5.68.13:8001', verbose=False)
    #分配输入输出变量
    outputs = [
    grpcclient.InferRequestedOutput('output0'),
    ]
    inputs = [grpcclient.InferInput('images', img.shape, 'FP32')]
    #将预处理完的图片赋值给inputs
    inputs[0].set_data_from_numpy(img)
    #启动流，（同样只需算法启动时调用一次）传入回调函数及用户数据，可以理解为将这个函数传给triton服务，triton完成推理后通过user_data对象成员函数中的那个队列将结果传回来，我们只需要从那个队列取结果即可。
    triton_client.start_stream(partial(completion_callback, user_data))
    start_time = time.time()
    n=1000
    for i in range(n): #这个for和下面的while是测试用的，使用时不是这么写
        #调用triton_client对象发起异步推理，可以理解为通过stream把输入数据传给
        #triton，我们在使用时可在另一个进程或线程中通过user_data对象中的那个队列获取triton推理完
        #的结果
        triton_client.async_stream_infer(
        'scsmodel',
        inputs,
        request_id=str(i),
        model_version='',
        outputs=outputs)
    while True:
        (results, error) = user_data._completed_requests.get()
        result = results.get_response(True)
        print(result['id'])
        print(results.as_numpy('output0'))
        if int(result['id'])== n-1:
            end_time = time.time()
            print("cost time:",end_time - start_time)

'''
import cv2
import time
import numpy as np
import tritonclient.grpc as grpcclient
if __name__ == '__main__':
    imgpath = '/mnt/zj/datasets/prepare_dataset/steel_test/c14fa2a403b5f87ceb934a1de43f2ea2.jpg'
    # 预处理
    img0 = cv2.imread(imgpath)
    img = cv2.resize(img0,(640,640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32)
    img /= 255.0
    img = np.stack([img],0) #可多张图片，这个list里弄成多张图就行
    # 初始化triton客户端对象
    n=1000
    start_time = time.time()
    for i in range(n):
        triton_client = grpcclient.InferenceServerClient(
        url='10.5.68.13:8001', verbose=False) #ip地址填本机ip地址，端口直接
        8001
        outputs = [
        grpcclient.InferRequestedOutput('output0'),
        ]
        inputs = [grpcclient.InferInput('images', img.shape, 'FP32')]
        inputs[0].set_data_from_numpy(img)
        result = triton_client.infer(
        'scsmodel',
        inputs,
        request_id=str(''),
        model_version='',
        outputs=outputs)
        #拿到结果
        output_array = result.as_numpy('output0')
        print(len(output_array[0]))
    end_time = time.time()
    print("cost time:",end_time - start_time)
'''