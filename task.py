# -*-coding:GBK -*-
import time
import queue
import threading
import argparse
from algrithom import *
from model.model_infer.tools.parser import get_config,load_config_from_file
import cv2
import os
from algrithom.tool.logger import get_logger
import subprocess
import numpy as np
from typing import Optional
import select

save_log_path = os.path.join('data/logs', "alg_log.txt")
logger = get_logger(save_log_path,'taskinfo')
# logger.info('*' * 50)
# 全局退出事件

stop_event = threading.Event()


fps=25

def clear_half_queue(q):
    # 获取队列的大小
    size = q.qsize()
    # 计算要丢弃的元素数量
    discard_count = size // 2

    # 丢弃一半的数据
    for _ in range(discard_count):
        try:
            q.get_nowait()  # 使用 get_nowait() 避免不必要的阻塞
        except queue.Empty:
            # 在理论上，这里不应该发生，因为我们已经检查了队列的大小
            # 但由于多线程环境的不确定性，最好还是处理一下这个异常
            break

class FFmpegStreamReader:
    def __init__(
            self,
            source: str,
            frame_size: tuple = (640, 480),
            fps: int = 25,
            read_timeout: int = 10,
            reconnect_interval: int = 5,
            max_retries: int = 10
    ):
        """
        基于FFmpeg的视频流读取器
        :param source: 视频源地址（支持rtsp/rtmp/http/file等）
        :param frame_size: 期望输出帧尺寸 (width, height)
        :param fps: 期望输出帧率
        :param read_timeout: 读取超时时间（秒）
        :param reconnect_interval: 重连间隔（秒）
        :param max_retries: 最大重试次数
        """
        self.source = source
        self.width, self.height = frame_size
        self.fps = fps
        self.read_timeout = read_timeout
        self.reconnect_interval = reconnect_interval
        self.max_retries = max_retries

        self._process: Optional[subprocess.Popen] = None
        self._frame_queue = queue.Queue(maxsize=10)
        self._running = threading.Event()
        self._retry_count = 0
        self._current_frame = None
        self._current_meta=None

        # FFmpeg参数配置
        self._ffmpeg_cmd = [
            'ffmpeg',
            '-re',  # 关键修复：按帧率读取本地文件
            '-hide_banner',
            '-loglevel', 'error',
            '-max_delay', '500000',
            '-i', self.source,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            'pipe:1'
        ]
        # 初始化时间相关参数
        self.stream_start = None
        self.frame_count = 0
        self._frame_queue = queue.Queue(maxsize=100)  # 存储元组(frame, metadata)

    def _start_process(self) -> bool:
        try:
            self._process = subprocess.Popen(
                self._ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # 禁用Python缓冲
            )
            # 立即检查错误输出
            time.sleep(0.1)
            # err = self._process.stderr.read()
            # if err:
            #     logger.error(f"FFmpeg启动错误: {err.decode()}")
            #     return False
            return True
        except Exception as e:
            logger.info(f"进程启动失败: {str(e)}")
            return False
    def _read_frames(self):
        """增强版帧读取线程"""
        self.stream_start = time.time()
        frame_bytes = self.width * self.height * 3
        buffer = bytearray()
        read_retries = 0
        max_read_retries = 5  # 最大连续读取重试次数
        health_check_interval = 10  # 健康检查间隔（秒）
        last_health_check = time.time()

        while self._running.is_set():
            try:
                # 非阻塞读取（增加重试机制）
                chunk = b''
                for _ in range(3):  # 最多重试3次
                    rlist, _, _ = select.select([self._process.stdout], [], [], 0.5)
                    if rlist:
                        chunk = self._process.stdout.read(frame_bytes)
                        # logger.debug(f"收到数据块大小: {len(chunk)} bytes")
                        if chunk:
                            break
                        else:
                            logger.info("收到空数据块，重试中...")
                    else:
                        logger.info("等待数据超时，重试中...")
                        time.sleep(0.2)

                # 处理空数据情况
                if not chunk:
                    if self.source.startswith('file'):
                        logger.info("文件读取结束，自动重启")
                        self._restart()
                        continue
                    read_retries += 1
                    if read_retries > max_read_retries:
                        logger.info(f"连续{max_read_retries}次读取失败，准备重连")
                        self._restart()
                        read_retries = 0
                        buffer = bytearray()
                    continue
                buffer.extend(chunk)
                # logger.info(f"收到数据块：{len(chunk)}字节，缓冲总量：{len(buffer)}字节")

                # 帧提取逻辑（增强容错）
                while len(buffer) >= frame_bytes:
                    logger.info(f"从缓冲区提取图片")
                    raw_frame = bytes(buffer[:frame_bytes])
                    del buffer[:frame_bytes]

                    # # 帧完整性校验（新增头尾校验）
                    # if not self._validate_frame(raw_frame):
                    #     logger.info("帧数据校验失败，可能存在数据损坏")
                    #     continue

                    # 转换和处理帧数据
                    try:
                        frame = np.frombuffer(raw_frame, dtype=np.uint8)
                        frame = frame.reshape((self.height, self.width, 3))
                    except ValueError as e:
                        logger.info(f"帧转换异常：{str(e)}")
                        continue

                    # 生成增强元数据
                    self.frame_count += 1
                    metadata = {
                        'frame_number': self.frame_count,
                        'calculated_ts': self.stream_start + (self.frame_count / self.fps),
                        'received_ts': time.time(),
                        'source_duration': self.frame_count / self.fps,
                        'buffer_level': len(buffer),
                        # 'stream_health': self._calculate_stream_health()
                    }

                    # 安全放入队列
                    try:

                        self._frame_queue.put_nowait((frame, metadata))
                    except queue.Full:
                        logger.info("队列已满，执行智能清理")
                        self._smart_queue_clean()
                        try:
                            self._frame_queue.put_nowait((frame, metadata))
                        except queue.Full:
                            logger.info("紧急清理后仍无法放入队列，丢弃关键帧")

                # 实时错误监控
                self._monitor_ffmpeg_errors()

            except (IOError, OSError) as e:
                logger.info(f"系统级IO异常：{str(e)}")
                self._restart()
            except Exception as e:
                logger.info(f"未处理异常：{str(e)}", exc_info=True)
                self._running.clear()

    def _process_alive(self):
        """检查FFmpeg进程状态（增强版）"""
        if self._process is None:
            return False
        return self._process.poll() is None

    def _validate_frame(self, data):
        """帧数据校验（示例校验头尾）"""
        # 头部校验：前10像素平均值应在合理范围
        header = np.frombuffer(data[:30], dtype=np.uint8)  # 前10个像素的BGR值
        if np.mean(header) < 5 or np.mean(header) > 250:
            return False

        # 尾部校验：最后4字节作为简单校验和
        checksum = sum(data[-4:]) % 256
        if checksum != 0:  # 示例校验逻辑，实际需根据数据特性调整
            return False

        return True

    def _calculate_stream_health(self):
        """计算流健康度指标"""
        health = 100
        # 根据缓冲水平扣分
        health -= min(len(self.buffer) // 1024, 50)  # 每KB扣1分，最多扣50分
        # 根据延迟扣分
        delay = time.time() - self.stream_start - (self.frame_count / self.fps)
        health -= min(int(delay * 10), 30)  # 每秒延迟扣10分，最多扣30分
        return max(health, 0)

    def _smart_queue_clean(self):
        """智能队列清理策略"""
        try:
            current_size = self._frame_queue.qsize()
            if current_size > self._frame_queue.maxsize // 2:
                # 保留最新50%的数据
                keep_count = current_size // 2
                temp_list = []
                for _ in range(keep_count):
                    temp_list.append(self._frame_queue.get_nowait())
                self._frame_queue.queue.clear()
                for item in temp_list[-keep_count:]:
                    self._frame_queue.put_nowait(item)
                logger.info(f"执行智能清理，保留最近{keep_count}帧")
        except Exception as e:
            logger.info(f"队列清理失败：{str(e)}")

    def _monitor_ffmpeg_errors(self):
        """实时错误监控"""
        while True:
            rlist, _, _ = select.select([self._process.stderr], [], [], 0)
            if not rlist:
                break

            err_line = self._process.stderr.readline().decode()
            if err_line:
                logger.info(f"FFmpeg错误：{err_line.strip()}")
                # 关键错误立即重启
                if "Connection refused" in err_line or "Server returned 404" in err_line:
                    self._restart()


    def _clear_old_frames(self):
        """清理队列旧数据"""
        try:
            for _ in range(self._frame_queue.qsize() // 2):
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass

    def _restart(self):
        """增强的重启方法"""
        logger.info("执行重启流程...")
        self._stop_process()

        # 指数退避重试
        retry_delay = min(2 ** self._retry_count, 30)
        time.sleep(retry_delay)

        if self._start_process():
            self._retry_count = 0
            self.stream_start = time.time()
            self.frame_count = 0
        else:
            self._retry_count += 1
            if self._retry_count >= self.max_retries:
                logger.info("达到最大重试次数，停止采集")
                self._running.clear()

    def _stop_process(self):
        """停止FFmpeg进程"""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            finally:
                self._process = None

    def start(self):
        """启动采集线程"""
        if not self._running.is_set():
            if self._start_process():
                self._running.set()
                threading.Thread(target=self._read_frames, daemon=True).start()

    def stop(self):
        """停止采集"""
        self._running.clear()
        self._stop_process()

    def read(self) -> Optional[tuple]:
        try:
            while not self._frame_queue.empty():
                self._current_frame, self._current_meta = self._frame_queue.get_nowait()
            return self._current_frame, self._current_meta
        except queue.Empty:
            return self._current_frame, self._current_meta
def decode(srcQueue, url: str, retry_interval: int = 30,max_retries=100):
    reader = FFmpegStreamReader(
        source=url,
        frame_size=(1920, 1080),
        fps=25,
        read_timeout=30,
        reconnect_interval=retry_interval,
        max_retries=max_retries
    )
    # 启动采集
    reader.start()
    try:
        count=0
        while True:
            frame, meta = reader.read()
            if frame is not None:
                count += 1
                # 此处添加图像处理逻辑
                if count % 5 != 0:
                    continue
                frame=frame.copy()
                frame.flags.writeable = True
                timestamp=meta['calculated_ts']

                msg_format = {"picture": frame, "camera_id": 'cam_id', "timestamp": timestamp, "frame_number": count}
            # time.sleep(0.03)  # 模拟处理间隔
                try:
                    srcQueue.put(msg_format, block=True, timeout=1)
                    logger.info("图片入列成功:{}".format(count))
                except queue.Full:
                    logger.info("srcQueue已满，无法添加更多数据。")
                    clear_half_queue(srcQueue)
                except Exception as e:
                    logger.info("解码过程中发生错误: %s:{}".format(e))
    except KeyboardInterrupt:
        reader.stop()
        print("采集已停止")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/home/ww/work/project/triton_project/config/lifejacket_algo_config.yaml', help='config path')
    opt = parser.parse_args()
    paramdic = get_config()
    paramdic.merge_from_file(opt.cfg)
    triton_config_path='model/model_infer/config/triton_config.yaml'
    triton_cfg = get_config()
    triton_cfg.merge_from_file(triton_config_path)
    paramdic['triton_cfg'] = triton_cfg
    if paramdic['trackerType']=='bytetrack':
        tracker_cfg_path='model/trackers/cfg/bytetrack.yaml'
    else:
        tracker_cfg_path='model/trackers/cfg/botsort.yaml'
    paramdic['tracker_cfg'] = load_config_from_file(tracker_cfg_path)
    url = paramdic.videosTask.videosId
    srcQueue = queue.Queue(maxsize=30)

    paramdic['pictureQueue'] = srcQueue

    if paramdic.algorithmType == 'smokefiredetect':
        paramdic["model_name"] = 'smokefiremodel'
        algorithm = SmokefireAlgThread
    elif paramdic.algorithmType == 'fencedetect':
        paramdic["model_name"] = 'fencemodel'
        algorithm = FenceAlgThread
    elif paramdic.algorithmType=='helmetdetect':
        paramdic["model_name"] = 'wearmodel'
        algorithm = HelmetAlgThread
    elif paramdic.algorithmType=='safebeltdetect':
        paramdic["model_name"] = 'wearmodel'
        algorithm = SafebeltAlgThread
    elif paramdic.algorithmType=='vasdetect':
        paramdic["model_name"] = 'scsmodel'
        algorithm = VasAlgThread
    elif paramdic.algorithmType=='crosslinedetect':
        paramdic["model_name"] = 'scsmodel'
        algorithm = CrosslineAlgThread
    elif paramdic.algorithmType=='mechinedetect':
        paramdic["model_name"] = 'scsmodel'
        algorithm = MechineAlgThread
    elif paramdic.algorithmType=='crowdcountdetect':
        paramdic["model_name"] = 'scsmodel'
        algorithm = CrowdcountAlgThread
    elif paramdic.algorithmType=='reflectivevestdetect':
        paramdic["model_name"] = 'wearmodel'
        algorithm = FlectivevestAlgThread
    elif paramdic.algorithmType=='lifejacketdetect':
        paramdic["model_name"] = 'wearmodel'
        algorithm = LifejacketAlgThread
    else:
        print('算法类型字段错误')
    resQueue = queue.Queue(maxsize=500)
    paramdic['resultQueue'] = resQueue
    decodeThread = threading.Thread(target=decode, args=(srcQueue, url))
    decodeThread.start()

    AlgThread = threading.Thread(target=algorithm, args=(paramdic,))
    AlgThread.daemon = True
    AlgThread.start()

