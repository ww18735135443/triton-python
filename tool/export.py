import torch
import onnx

def convert_pth_to_onnx(pth_file, onnx_file, input_size=(1, 3, 224, 224)):
    """
    Convert a .pth model to .onnx format.

    Args:
        pth_file (str): Path to the .pth model file.
        onnx_file (str): Path to the output .onnx file.
        input_size (tuple): Input size of the model (default: (1, 3, 224, 224)).
    """
    # 加载模型
    model = torch.load(pth_file, map_location=torch.device('cpu'))

    # 如果模型是状态字典，则创建一个新的模型实例并加载权重
    if 'state_dict' in model:
        # 假设模型是 ResNet18，根据实际情况替换
        model = models.resnet18()
        model.load_state_dict(model['state_dict'])
    else:
        model = model  # 直接使用模型

    # 设置模型为评估模式
    model.eval()

    # 创建一个示例输入张量
    dummy_input = torch.randn(input_size, requires_grad=True)

    # 导出模型到 ONNX 格式
    torch.onnx.export(
        model,  # 模型
        dummy_input,  # 示例输入
        onnx_file,  # 输出文件名
        export_params=True,  # 存储训练好的参数
        opset_version=10,  # ONNX 版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 输入节点名称
        output_names=['output'],  # 输出节点名称
        dynamic_axes={'input': {0: 'batch_size'},  # 动态轴
                      'output': {0: 'batch_size'}}
    )

    # 检查 ONNX 模型是否有效
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型验证成功！")

# 示例用法
pth_file = 'model.pth'
onnx_file = 'model.onnx'

convert_pth_to_onnx(pth_file, onnx_file)
