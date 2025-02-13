## <div align="center">项目部署说明</div>
<details>
  <summary>项目准备</summary>

### 项目地址
项目位置：10.5.68.11：/home/xqw/project/triton_project/
gitlab代码仓库：http://10.5.55.26/xieqiwang/alg-deploy.git

### 环境准备
安装
mysql
ffmpeg
### 启动triton镜像
拉取triton镜像nvcr.io/nvidia/tritonserver:23.10-py3
```bash
#拉取镜像
docker pull nvcr.io/nvidia/tritonserver:23.10-py3
#启动triton
docker run --gpus all  ---restart=always -rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/xqw/project/triton_project/model_repository/:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models --model-control-mode poll
```
### python环境准备
根据requirement.yaml配置python环境
```bash
#进入项目
cd /home/xqw/project/triton_project/
# 激活环境
conda activate /home/xqw/condaenv/yolo
```

### 启动接口服务
```bash
#视频分析任务接口
nohup python task-flask2.py >task-analyse.log 2>&1 &
# 算法集市接口
nohup python imgflask.py nohup python imgflask.py > alg_mart.log 2>&1 &
```