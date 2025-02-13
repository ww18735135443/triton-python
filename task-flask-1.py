import os
import yaml
import subprocess
import signal
import uuid
import time
import json
from flask import Flask, request, jsonify
import psutil
import select
import atexit

def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

import signal


def run_task(cfg_path):
    # Simulate a long-running task
    # Replace with your actual long-running task command
    command = ['python', 'task.py', '--cfg', cfg_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Lists to hold the output and error lines
    stdout_lines = []
    stderr_lines = []

    # Function to read from a stream in a non-blocking way
    def read_stream(stream, lines_list, timeout):
        last_time = time.time()
        while True:
            # Use select to check if there is data to read without blocking
            readable, _, _ = select.select([stream], [], [], timeout)
            if not readable:
                # Timeout reached, no more data to read
                break
            # Read a line from the stream
            line = stream.readline()
            if not line:
                # End of stream reached
                break
            # Decode the line and add it to the list
            lines_list.append(line.decode('utf-8').strip())
            # Update the last time data was read to reset the timeout
            last_time = time.time()
        # If the stream is closed and we've read all data, return True
        return stream.closed and not select.select([stream], [], [], 0)[0]

    # Start reading from both stdout and stderr
    stdout_done = read_stream(process.stdout, stdout_lines, timeout=2)
    stderr_done = read_stream(process.stderr, stderr_lines, timeout=2)

    # Check if the process is still running after the timeout
    still_running = process.poll() is None

    # Combine stdout and stderr lines into a single string for each
    stdout_str = '\n'.join(stdout_lines)
    stderr_str = '\n'.join(stderr_lines)

    # Prepare the result dictionary
    result = {
        'process': process,
        'success': still_running,  # Success if the process is not still running
        'stdout': stdout_str,
        'stderr': stderr_str
    }

    # Optionally, if you want to kill the process if it's still running, you can do:
    # if still_running:
    #     process.terminate()

    return result


def get_interface_id_from_yaml(yaml_file_path):
    # 读取YAML文件
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    # 获取interfaceId的值
    interface_id = data.get('interfaceId')
    return interface_id


def terminate_all_tasks():
    for task in tasks.values():
        process = task['process']
        if process.poll() is None:  # 如果进程还在运行
            process.terminate()  # 终止进程
            process.wait()  # 等待进程终止
atexit.register(terminate_all_tasks)  # 注册清理函数
app = Flask(__name__)
tasks = {}
task_file='data/task-file'
@app.route('/recover/<single>', methods=['GET'])
def recover_task(single):
    if single:
        task_file_dir='data/task-file'
    recovery_results = []  # 创建列表来存储恢复结果
    task_yaml_files = os.listdir(task_file_dir)
    for task_file in task_yaml_files :
        yaml_file_path=os.path.join(task_file_dir, task_file)
        unique_id=get_interface_id_from_yaml(yaml_file_path)
        if unique_id in tasks:
            task = tasks[unique_id]
            process = task['process']
            if process.poll() is None:
                p = psutil.Process(process.pid)  # 假设 process 对象有一个 pid 属性
                # 获取资源占用情况
                cpu_percent = p.cpu_percent(interval=1.0)  # CPU 使用率，interval 表示采样时间间隔
                if cpu_percent>100:
                    recovery_results.append({'unique_id': unique_id, 'status': 'already_running'})
                    continue
        process_result = run_task(yaml_file_path)
        process=process_result['process']
        stdout=process_result['stdout']
        stderr=process_result['stderr']
        print(process_result)
        time.sleep(3)
        if process.poll() is not None or stderr:
            os.remove(yaml_file_path)
            print("{} task recover fail,stdout:{},stderr:{}".format(unique_id,stdout,stderr))
            recovery_results.append({
                'unique_id': unique_id,
                'status': 'fail',
                'stdout': stdout,
                'stderr': stderr
            })
        else:
            is_running = process.poll() is None
            if is_running:# and cpu_percent>5:
                tasks[unique_id] = {'process': process, 'status':True, 'yaml_file': yaml_file_path}
                print("{} task recover success,stdout:{},stderr:{}".format(unique_id,stdout,stderr))
                recovery_results.append({
                    'unique_id': unique_id,
                    'status': 'success',
                    'stdout': stdout,
                    'stderr': stderr  # 通常成功时 stderr 为空，但保留以防万一
                })
            else:
                os.remove(yaml_file_path)
                print("{} task recover fail,stdout:{},stderr:{}".format(unique_id,stdout,stderr))
                recovery_results.append({
                    'unique_id': unique_id,
                    'status': 'unexpected_fail',  # 意外失败，理论上不应该发生
                    'stdout': stdout,
                    'stderr': stderr
                })
    return jsonify(recovery_results), 202

@app.route('/start', methods=['POST'])
def start_task():
    data = request.json
    if data['interfaceId']:
        unique_id=data['interfaceId']
        if unique_id in tasks:
            return jsonify({'error': 'Task existed'}), 400
    else:
        unique_id = str(uuid.uuid4())
    data['interfaceId']=unique_id
    yaml_file_path = f"{task_file}/{unique_id}.yaml"
    save_yaml(data, yaml_file_path)

    process_result = run_task(yaml_file_path)
    process=process_result['process']
    stdout=process_result['stdout']
    stderr=process_result['stderr']
    print(process_result)
    time.sleep(3)
    if process.poll() is not None or stderr:
        os.remove(yaml_file_path)
        return jsonify({'error': unique_id,'error':stderr}), 400
    else:
        # p = psutil.Process(process.pid)  # 假设 process 对象有一个 pid 属性

        # 获取进程状态
        is_running = process.poll() is None

        # 获取资源占用情况
        # cpu_percent = p.cpu_percent(interval=1.0)  # CPU 使用率，interval 表示采样时间间隔
        if is_running:# and cpu_percent>5:
            tasks[unique_id] = {'process': process, 'status':True,'yaml_file': yaml_file_path}
            return jsonify({'success': unique_id,'info':stdout}), 202
        else:
            os.remove(yaml_file_path)
            return jsonify({'error': unique_id,'suggest':'please check video address is correct'}), 400

@app.route('/<id>/status', methods=['GET'])
def get_status(id):
    if id not in tasks:
        return jsonify({'error': 'Task not found'}), 400

    task = tasks[id]
    process = task['process']
    # 使用 psutil 获取进程信息
    try:
        p = psutil.Process(process.pid)  # 假设 process 对象有一个 pid 属性

        # 获取进程状态
        is_running = process.poll() is None

        # 获取资源占用情况
        cpu_percent = p.cpu_percent(interval=1.0)  # CPU 使用率，interval 表示采样时间间隔
        memory_info = p.memory_info()  # 内存使用情况
        memory_percent = p.memory_percent()  # 内存使用率

        # 构建响应数据
        response_data = {
            'id': id,
            'pid':process.pid,
            'status': is_running,
            'yaml_file': task['yaml_file'],
            'resources': {
                'cpu_percent': cpu_percent,
                'memory_info': {
                    'rss': memory_info.rss,  # 常驻集大小
                    'vms': memory_info.vms,  # 虚拟内存大小
                },
                'memory_percent': memory_percent
            }
        }

        return jsonify(response_data)

    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        # 处理可能的异常，如进程不存在、访问被拒绝或僵尸进程
        return jsonify({'error': 'Unable to retrieve process information'}), 400

@app.route('/<id>/stop', methods=['GET'])
def stop_task(id):
    if id not in tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = tasks[id]
    process = task['process']
    os.kill(process.pid, signal.SIGTERM)  # Send SIGTERM to terminate the process
    process.wait()  # Wait for the process to terminate

    # #remove the task from the tasks dictionary
    # del tasks[id]

    return jsonify({'id': id, 'status': 'stopped'})

@app.route('/<id>/restart', methods=['GET'])
def restart_task(id):
    if id in tasks:
        return jsonify({'error': 'Task already in running,please stop then restart'}), 200
    if os.path.exists(f"{task_file}/{id}.yaml"):
        yaml_file_path=f"{task_file}/{id}.yaml"
    # Start a new process
    new_process = run_task(yaml_file_path)
    tasks[id] = {'process': new_process, 'yaml_file': yaml_file_path}
    time.sleep(5)
    if new_process.poll() is not None:
        return jsonify({'error': id}), 400
    else:
        return jsonify({'success': id}), 202

@app.route('/<id>/delete', methods=['GET'])
def delete_task(id):
    if id not in tasks:
        if os.path.exists(f"{task_file}/{id}.yaml"):
            os.remove(f"{task_file}/{id}.yaml")
            return jsonify({'success': 'task deleted '}), 202
    else:
        task = tasks[id]
        process = task['process']

        # Stop the process if it's still running
        if process.poll() is None:
            os.kill(process.pid, signal.SIGTERM)
            process.wait()

        # Remove the YAML file
        os.remove(task['yaml_file'])

        # Remove the task from the tasks dictionary
        del tasks[id]

        return jsonify({'id': id, 'status': 'process be kill and task be deleted'})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=29001)