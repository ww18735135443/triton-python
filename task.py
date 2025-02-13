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
# ȫ���˳��¼�

stop_event = threading.Event()


fps=25

def clear_half_queue(q):
    # ��ȡ���еĴ�С
    size = q.qsize()
    # ����Ҫ������Ԫ������
    discard_count = size // 2

    # ����һ�������
    for _ in range(discard_count):
        try:
            q.get_nowait()  # ʹ�� get_nowait() ���ⲻ��Ҫ������
        except queue.Empty:
            # �������ϣ����ﲻӦ�÷�������Ϊ�����Ѿ�����˶��еĴ�С
            # �����ڶ��̻߳����Ĳ�ȷ���ԣ���û��Ǵ���һ������쳣
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
        ����FFmpeg����Ƶ����ȡ��
        :param source: ��ƵԴ��ַ��֧��rtsp/rtmp/http/file�ȣ�
        :param frame_size: �������֡�ߴ� (width, height)
        :param fps: �������֡��
        :param read_timeout: ��ȡ��ʱʱ�䣨�룩
        :param reconnect_interval: ����������룩
        :param max_retries: ������Դ���
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

        # FFmpeg��������
        self._ffmpeg_cmd = [
            'ffmpeg',
            '-re',  # �ؼ��޸�����֡�ʶ�ȡ�����ļ�
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
        # ��ʼ��ʱ����ز���
        self.stream_start = None
        self.frame_count = 0
        self._frame_queue = queue.Queue(maxsize=100)  # �洢Ԫ��(frame, metadata)

    def _start_process(self) -> bool:
        try:
            self._process = subprocess.Popen(
                self._ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # ����Python����
            )
            # �������������
            time.sleep(0.1)
            # err = self._process.stderr.read()
            # if err:
            #     logger.error(f"FFmpeg��������: {err.decode()}")
            #     return False
            return True
        except Exception as e:
            logger.info(f"��������ʧ��: {str(e)}")
            return False
    def _read_frames(self):
        """��ǿ��֡��ȡ�߳�"""
        self.stream_start = time.time()
        frame_bytes = self.width * self.height * 3
        buffer = bytearray()
        read_retries = 0
        max_read_retries = 5  # ���������ȡ���Դ���
        health_check_interval = 10  # ������������룩
        last_health_check = time.time()

        while self._running.is_set():
            try:
                # ��������ȡ���������Ի��ƣ�
                chunk = b''
                for _ in range(3):  # �������3��
                    rlist, _, _ = select.select([self._process.stdout], [], [], 0.5)
                    if rlist:
                        chunk = self._process.stdout.read(frame_bytes)
                        # logger.debug(f"�յ����ݿ��С: {len(chunk)} bytes")
                        if chunk:
                            break
                        else:
                            logger.info("�յ������ݿ飬������...")
                    else:
                        logger.info("�ȴ����ݳ�ʱ��������...")
                        time.sleep(0.2)

                # ������������
                if not chunk:
                    if self.source.startswith('file'):
                        logger.info("�ļ���ȡ�������Զ�����")
                        self._restart()
                        continue
                    read_retries += 1
                    if read_retries > max_read_retries:
                        logger.info(f"����{max_read_retries}�ζ�ȡʧ�ܣ�׼������")
                        self._restart()
                        read_retries = 0
                        buffer = bytearray()
                    continue
                buffer.extend(chunk)
                # logger.info(f"�յ����ݿ飺{len(chunk)}�ֽڣ�����������{len(buffer)}�ֽ�")

                # ֡��ȡ�߼�����ǿ�ݴ�
                while len(buffer) >= frame_bytes:
                    logger.info(f"�ӻ�������ȡͼƬ")
                    raw_frame = bytes(buffer[:frame_bytes])
                    del buffer[:frame_bytes]

                    # # ֡������У�飨����ͷβУ�飩
                    # if not self._validate_frame(raw_frame):
                    #     logger.info("֡����У��ʧ�ܣ����ܴ���������")
                    #     continue

                    # ת���ʹ���֡����
                    try:
                        frame = np.frombuffer(raw_frame, dtype=np.uint8)
                        frame = frame.reshape((self.height, self.width, 3))
                    except ValueError as e:
                        logger.info(f"֡ת���쳣��{str(e)}")
                        continue

                    # ������ǿԪ����
                    self.frame_count += 1
                    metadata = {
                        'frame_number': self.frame_count,
                        'calculated_ts': self.stream_start + (self.frame_count / self.fps),
                        'received_ts': time.time(),
                        'source_duration': self.frame_count / self.fps,
                        'buffer_level': len(buffer),
                        # 'stream_health': self._calculate_stream_health()
                    }

                    # ��ȫ�������
                    try:

                        self._frame_queue.put_nowait((frame, metadata))
                    except queue.Full:
                        logger.info("����������ִ����������")
                        self._smart_queue_clean()
                        try:
                            self._frame_queue.put_nowait((frame, metadata))
                        except queue.Full:
                            logger.info("������������޷�������У������ؼ�֡")

                # ʵʱ������
                self._monitor_ffmpeg_errors()

            except (IOError, OSError) as e:
                logger.info(f"ϵͳ��IO�쳣��{str(e)}")
                self._restart()
            except Exception as e:
                logger.info(f"δ�����쳣��{str(e)}", exc_info=True)
                self._running.clear()

    def _process_alive(self):
        """���FFmpeg����״̬����ǿ�棩"""
        if self._process is None:
            return False
        return self._process.poll() is None

    def _validate_frame(self, data):
        """֡����У�飨ʾ��У��ͷβ��"""
        # ͷ��У�飺ǰ10����ƽ��ֵӦ�ں���Χ
        header = np.frombuffer(data[:30], dtype=np.uint8)  # ǰ10�����ص�BGRֵ
        if np.mean(header) < 5 or np.mean(header) > 250:
            return False

        # β��У�飺���4�ֽ���Ϊ��У���
        checksum = sum(data[-4:]) % 256
        if checksum != 0:  # ʾ��У���߼���ʵ��������������Ե���
            return False

        return True

    def _calculate_stream_health(self):
        """������������ָ��"""
        health = 100
        # ���ݻ���ˮƽ�۷�
        health -= min(len(self.buffer) // 1024, 50)  # ÿKB��1�֣�����50��
        # �����ӳٿ۷�
        delay = time.time() - self.stream_start - (self.frame_count / self.fps)
        health -= min(int(delay * 10), 30)  # ÿ���ӳٿ�10�֣�����30��
        return max(health, 0)

    def _smart_queue_clean(self):
        """���ܶ����������"""
        try:
            current_size = self._frame_queue.qsize()
            if current_size > self._frame_queue.maxsize // 2:
                # ��������50%������
                keep_count = current_size // 2
                temp_list = []
                for _ in range(keep_count):
                    temp_list.append(self._frame_queue.get_nowait())
                self._frame_queue.queue.clear()
                for item in temp_list[-keep_count:]:
                    self._frame_queue.put_nowait(item)
                logger.info(f"ִ�����������������{keep_count}֡")
        except Exception as e:
            logger.info(f"��������ʧ�ܣ�{str(e)}")

    def _monitor_ffmpeg_errors(self):
        """ʵʱ������"""
        while True:
            rlist, _, _ = select.select([self._process.stderr], [], [], 0)
            if not rlist:
                break

            err_line = self._process.stderr.readline().decode()
            if err_line:
                logger.info(f"FFmpeg����{err_line.strip()}")
                # �ؼ�������������
                if "Connection refused" in err_line or "Server returned 404" in err_line:
                    self._restart()


    def _clear_old_frames(self):
        """������о�����"""
        try:
            for _ in range(self._frame_queue.qsize() // 2):
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass

    def _restart(self):
        """��ǿ����������"""
        logger.info("ִ����������...")
        self._stop_process()

        # ָ���˱�����
        retry_delay = min(2 ** self._retry_count, 30)
        time.sleep(retry_delay)

        if self._start_process():
            self._retry_count = 0
            self.stream_start = time.time()
            self.frame_count = 0
        else:
            self._retry_count += 1
            if self._retry_count >= self.max_retries:
                logger.info("�ﵽ������Դ�����ֹͣ�ɼ�")
                self._running.clear()

    def _stop_process(self):
        """ֹͣFFmpeg����"""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            finally:
                self._process = None

    def start(self):
        """�����ɼ��߳�"""
        if not self._running.is_set():
            if self._start_process():
                self._running.set()
                threading.Thread(target=self._read_frames, daemon=True).start()

    def stop(self):
        """ֹͣ�ɼ�"""
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
    # �����ɼ�
    reader.start()
    try:
        count=0
        while True:
            frame, meta = reader.read()
            if frame is not None:
                count += 1
                # �˴����ͼ�����߼�
                if count % 5 != 0:
                    continue
                frame=frame.copy()
                frame.flags.writeable = True
                timestamp=meta['calculated_ts']

                msg_format = {"picture": frame, "camera_id": 'cam_id', "timestamp": timestamp, "frame_number": count}
            # time.sleep(0.03)  # ģ�⴦����
                try:
                    srcQueue.put(msg_format, block=True, timeout=1)
                    logger.info("ͼƬ���гɹ�:{}".format(count))
                except queue.Full:
                    logger.info("srcQueue�������޷���Ӹ������ݡ�")
                    clear_half_queue(srcQueue)
                except Exception as e:
                    logger.info("��������з�������: %s:{}".format(e))
    except KeyboardInterrupt:
        reader.stop()
        print("�ɼ���ֹͣ")

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
        print('�㷨�����ֶδ���')
    resQueue = queue.Queue(maxsize=500)
    paramdic['resultQueue'] = resQueue
    decodeThread = threading.Thread(target=decode, args=(srcQueue, url))
    decodeThread.start()

    AlgThread = threading.Thread(target=algorithm, args=(paramdic,))
    AlgThread.daemon = True
    AlgThread.start()

