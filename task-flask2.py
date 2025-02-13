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
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
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
    print(stdout_done)
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
def get_algtype_from_yaml(yaml_file_path):
    # 读取YAML文件
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    # 获取interfaceId的值
    algorithmType = data.get('algorithmType')
    return algorithmType

def kill_process_byid(id):
    try:
        task=tasks[id]
        process = task['process']
        p = psutil.Process(process.pid)
        # 尝试优雅地终止进程
        p.terminate()
        print(f"Process {p.pid} is being terminated gracefully.")
        # 等待进程终止，这里设置了一个超时时间，可以根据实际情况调整
        p.wait(timeout=5)  # 等待5秒
        print(f"Process {p.pid} has been terminated.")
    except psutil.TimeoutExpired:
        # 如果进程在指定时间内没有终止，可以强制终止它
        p.kill()
        print(f"Process {p.pid} was forcefully killed.")
def cpu_check(p):
    # 获取资源占用情况
    cpu_sum=0
    for i in range(5):
        cpu_percent = p.cpu_percent(interval=0.2)  # CPU 使用率，interval 表示采样时间间隔
        time.sleep(0.2)
        cpu_sum += cpu_percent
    return cpu_sum/5
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
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:1415850364@10.5.68.11:3307/flask_sql'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # 关闭信号追踪以提高性能

db = SQLAlchemy(app)

class TaskStatus(db.Model):
    __tablename__ = 'algtask_status'  # 指定数据库中的表名
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # 自增主键
    name = db.Column(db.String(80), unique=True, nullable=False)  # 任务名称，唯一且非空
    status = db.Column(db.String(80), nullable=False)  # 任务状态，非空
    algtype = db.Column(db.String(80), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)  # 任务创建时间，默认为当前时间
    modify_time = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 任务修改时间，默认为当前时间，并在记录更新时自动更新

    def __repr__(self):
        return f'<task:%s,TaskStatus:%s,start time:%s,modify_time:%s >'%(self.name,self.status,self.start_time,self.modify_time)

# 创建数据库和表
with app.app_context():
    db.create_all()
@app.route('/recover/<single>', methods=['GET'])
def recover_task(single):
    if single:
        task_file_dir='data/task-file'
    recovery_results = []  # 创建列表来存储恢复结果
    task_yaml_files = os.listdir(task_file_dir)
    for task_file in task_yaml_files :
        yaml_file_path=os.path.join(task_file_dir, task_file)
        unique_id=get_interface_id_from_yaml(yaml_file_path)
        algtype=get_algtype_from_yaml(yaml_file_path)
        db_task = TaskStatus.query.filter_by(name=unique_id).first()
        if db_task is not None:
            if db_task.status == 'stopped' or db_task.status == 'delete':
                continue
        if unique_id in tasks:
            task = tasks[unique_id]
            if task['status'] == 'stopped':
                recovery_results.append({'unique_id': unique_id, 'status': 'already stopped,not need recover'})
                if db_task is None:
                    new_task = TaskStatus(
                        name=unique_id,
                        status='stopped',
                        algtype=algtype,
                        start_time=datetime.utcnow()
                    )
                    db.session.add(new_task)
                    db.session.commit()  # 提交事务到数据库
                continue
            process = task['process']
            if process.poll() is None:
                p = psutil.Process(process.pid)  # 假设 process 对象有一个 pid 属性
                # 获取资源占用情况
                cpu_percent = cpu_check(p)#p.cpu_percent(interval=1.0)  # CPU 使用率，interval 表示采样时间间隔
                if cpu_percent>100:
                    recovery_results.append({'unique_id': unique_id, 'status': 'already_running'})
                    if db_task is None:
                        new_task = TaskStatus(
                            name=unique_id,  #
                            status='running',
                            algtype=algtype,
                            start_time=datetime.utcnow()  # 可以省略，因为已有默认值
                        )
                        db.session.add(new_task)
                        db.session.commit()  # 提交事务到数据库
                    continue
                else:
                    try:
                        # 尝试优雅地终止进程
                        p.terminate()
                        print(f"Process {p.pid} is being terminated gracefully.")
                        # 等待进程终止，这里设置了一个超时时间，可以根据实际情况调整
                        p.wait(timeout=5)  # 等待5秒
                        print(f"Process {p.pid} has been terminated.")
                    except psutil.TimeoutExpired:
                        # 如果进程在指定时间内没有终止，可以强制终止它
                        p.kill()
                        print(f"Process {p.pid} was forcefully killed.")

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
                if db_task is None:
                    new_task = TaskStatus(
                        name=unique_id,
                        status='started',
                        algtype=algtype,
                        start_time=datetime.utcnow()
                    )
                    db.session.add(new_task)
                    db.session.commit()  # 提交事务到数据库
                else:
                    db_task.status = 'started'
                    db_task.modify_time = datetime.utcnow()
                    db.session.commit()
            else:
                os.remove(yaml_file_path)
                print("{} task recover fail,stdout:{},stderr:{}".format(unique_id,stdout,stderr))
                recovery_results.append({
                    'unique_id': unique_id,
                    'status': 'unexpected_fail',  # 意外失败，理论上不应该发生
                    'stdout': stdout,
                    'stderr': stderr
                })
                if db_task is None:
                    new_task = TaskStatus(
                        name=unique_id,
                        status='abnormal',
                        algtype=algtype,
                        start_time=datetime.utcnow()
                    )
                    db.session.add(new_task)
                    db.session.commit()  # 提交事务到数据库
                else:
                    db_task.status = 'abnormal'
                    db_task.modify_time = datetime.utcnow()
                    db.session.commit()
    return jsonify(recovery_results), 200

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
    algtype=data['algorithmType']
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
            tasks[unique_id] = {'process': process, 'status':'started','yaml_file': yaml_file_path}
            #将任务保存到数据库中
            db_task = TaskStatus.query.filter_by(name=unique_id).first()
            if db_task is None:
                new_task = TaskStatus(
                    name=unique_id,  #
                    algtype=algtype,
                    status='started',
                    start_time=datetime.utcnow()  # 可以省略，因为已有默认值
                )
                db.session.add(new_task)
                db.session.commit()  # 提交事务到数据库
            else:
                db_task.status = 'started'
                db_task.modify_time = datetime.utcnow()
                db.session.commit()
            return jsonify({'success': unique_id,'info':stdout}), 200
        else:
            os.remove(yaml_file_path)
            return jsonify({'error': unique_id,'suggest':'please check video address is correct'}), 400

@app.route('/<id>/status', methods=['GET'])
def get_status(id):
    db_task = TaskStatus.query.filter_by(name=id).first()
    # if db_task is None:
    #     return jsonify({'error': 'Task not found'}), 400
    if id not in tasks:
        if db_task is None:
            return jsonify({'error': 'Task not found'}), 400
        else:
            return jsonify({'error': 'Task not running','status':db_task.status}), 200

    task = tasks[id]
    process = task['process']
    # 使用 psutil 获取进程信息
    try:
        p = psutil.Process(process.pid)  # 假设 process 对象有一个 pid 属性

        # 获取进程状态
        is_running = process.poll() is None

        # 获取资源占用情况
        # cpu_percent1 = p.cpu_percent(interval=0.5)  # CPU 使用率，interval 表示采样时间间隔
        # time.sleep(0.5)
        # cpu_percent2 = p.cpu_percent(interval=0.5)
        cpu_percent=cpu_check(p)
        memory_info = p.memory_info()  # 内存使用情况
        memory_percent = p.memory_percent()  # 内存使用率
        status='stopped'
        if is_running:
            if cpu_percent>10:
                status='running'
            else:
                status='pending'
        # 构建响应数据
        response_data = {
            'id': id,
            'pid':process.pid,
            'status': status,
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
        tasks[id]['status']=status
        db_task.status = status
        db.session.commit()
        return jsonify(response_data),200

    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        # 处理可能的异常，如进程不存在、访问被拒绝或僵尸进程
        db_task.status = 'abnormal'
        db.session.commit()
        return jsonify({'error': 'Unable to retrieve process information'}), 400

@app.route('/<id>/stop', methods=['GET'])
def stop_task(id):
    db_task = TaskStatus.query.filter_by(name=id).first()
    if id not in tasks:
        if db_task is None:
            return jsonify({'error': 'Task not found'}), 400
        else:
            if db_task.status == 'stopped':
                return jsonify({'success': 'Task not in running'}), 200

    task = tasks[id]
    process = task['process']
    if process.poll() is None:
        os.kill(process.pid, signal.SIGTERM)  # Send SIGTERM to terminate the process
        process.wait()  # Wait for the process to terminate
    tasks[id]['status']="stopped"
    #同步数据库
    db_task.status = "stopped"
    db_task.modify_time = datetime.utcnow()
    db.session.commit()
    return jsonify({'id': id, 'status': 'stopped'}),200

@app.route('/<id>/restart', methods=['GET'])
def restart_task(id):
    db_task = TaskStatus.query.filter_by(name=id).first()
    if id in tasks and os.path.exists(f"{task_file}/{id}.yaml"):
        if db_task is None:
            #将任务保存到数据库中

            new_task = TaskStatus(
                name=id,  # 你可能需要从数据中获取或设置默认值
                status='running',
                algtype=get_algtype_from_yaml(f"{task_file}/{id}.yaml"),
                start_time=datetime.utcnow()  # 可以省略，因为已有默认值
            )
            db.session.add(new_task)
            db.session.commit()  # 提交事务到数据库
        if tasks[id]['status'] == 'running':
            return jsonify({'success':'Task already in running,please stop then restart'}), 200
        else:
            try:
                task=tasks[id]
                process = task['process']
                if process.poll() is None:
                    p = psutil.Process(process.pid)
                    # 尝试优雅地终止进程
                    p.terminate()
                    print(f"Process {p.pid} is being terminated gracefully.")
                    # 等待进程终止，这里设置了一个超时时间，可以根据实际情况调整
                    p.wait(timeout=5)  # 等待5秒
                    print(f"Process {p.pid} has been terminated.")
            except :
                # 如果进程在指定时间内没有终止，可以强制终止它
                p.kill()
                print(f"Process {p.pid} was forcefully killed.")
    if os.path.exists(f"{task_file}/{id}.yaml"):
        yaml_file_path=f"{task_file}/{id}.yaml"
    # Start a new process
    new_process_result = run_task(yaml_file_path)
    process=new_process_result['process']
    stdout=new_process_result['stdout']
    stderr=new_process_result['stderr']
    time.sleep(5)
    if process.poll() is not None:
        # tasks[id] = {'process': process,'status':'abnormal', 'yaml_file': yaml_file_path}
        if db_task is not None:
            db_task.status = 'abnormal'
            db_task.modify_time = datetime.utcnow()
            db.session.commit()
        return jsonify({'error': id}), 400
    else:
        tasks[id] = {'process': process,'status':'restarted', 'yaml_file': yaml_file_path}
        if db_task is not None:
            db_task.status = 'started'
            db_task.modify_time = datetime.utcnow()
            db.session.commit()
        return jsonify({'success': id}), 200

@app.route('/<id>/delete', methods=['GET'])
def delete_task(id):
    db_task = TaskStatus.query.filter_by(name=id).first()
    if id not in tasks:
        if os.path.exists(f"{task_file}/{id}.yaml"):
            os.remove(f"{task_file}/{id}.yaml")
        if db_task is not None:
            db_task.status = 'delete'
            db_task.modify_time = datetime.utcnow()
            db.session.commit()
            return jsonify({'success': 'task deleted '}), 200
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
        if db_task is not None:
            db_task.status = 'delete'
            db_task.modify_time = datetime.utcnow()
            db.session.commit()
        return jsonify({'id': id, 'status': 'process be kill and task be deleted'}),200
@app.route('/checkstatus/<single>', methods=['GET'])
def checkstatus(single):
    if single:
        db_tasks = TaskStatus.query.all()
    for db_task in db_tasks:
        if db_task.status != 'stopped' and db_task.status != 'delete':
            unique_id=db_task.name
            if unique_id  in tasks:
                task = tasks[unique_id]
                process = task['process']
                # 使用 psutil 获取进程信息
                try:
                    p = psutil.Process(process.pid)  # 假设 process 对象有一个 pid 属性
                    # 获取进程状态
                    is_running = process.poll() is None
                    cpu_percent=cpu_check(p)
                    memory_info = p.memory_info()  # 内存使用情况
                    memory_percent = p.memory_percent()  # 内存使用率
                    status='stopped'
                    if is_running:
                        if cpu_percent>10:
                            status='running'
                        else:
                            status='pending'
                    tasks[unique_id]['status']=status
                    db_task.status = status
                    db.session.commit()

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # 处理可能的异常，如进程不存在、访问被拒绝或僵尸进程
                    db_task.status = 'abnormal'
                    db.session.commit()
    return jsonify({'success': 'update database success'}),200
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=29001)