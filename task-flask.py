import os
import yaml
import subprocess
import signal
import uuid
import time
import json
from flask import Flask, request, jsonify

app = Flask(__name__)
tasks = {}
task_file='data/task-file'
def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

def run_task(cfg_path):
    # Simulate a long-running task
    # Replace with your actual long-running task command
    command = ['python', 'task.py', '--cfg', cfg_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

@app.route('/start', methods=['POST'])
def start_task():
    data = request.json
    if data['interfaceId']:
        unique_id=data['interfaceId']
    else:
        unique_id = str(uuid.uuid4())
    data['interfaceId']=unique_id
    yaml_file_path = f"{task_file}/{unique_id}.yaml"
    save_yaml(data, yaml_file_path)

    process = run_task(yaml_file_path)
    tasks[unique_id] = {'process': process, 'yaml_file': yaml_file_path}
    time.sleep(3)
    if process.poll() is not None:
        return jsonify({'start error': unique_id}), 400
    else:
        return jsonify({'start success': unique_id}), 202

@app.route('/<id>/status', methods=['GET'])
def get_status(id):
    if id not in tasks:
        return jsonify({'error': 'Task not found'}), 400

    task = tasks[id]
    process = task['process']
    return jsonify({
        'id': id,
        'status': process.poll() is None,  # True if process is running, False if it has terminated
        'yaml_file': task['yaml_file']
    })

@app.route('/<id>/stop', methods=['GET'])
def stop_task(id):
    if id not in tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = tasks[id]
    process = task['process']
    os.kill(process.pid, signal.SIGTERM)  # Send SIGTERM to terminate the process
    process.wait()  # Wait for the process to terminate

    #remove the task from the tasks dictionary
    del tasks[id]

    return jsonify({'id': id, 'status': 'stopped'})

@app.route('/<id>/restart', methods=['GET'])
def restart_task(id):
    if id in tasks:
        return jsonify({'error': 'Task already in running'}), 200
    if os.path.exists(f"{task_file}/{id}.yaml"):
        yaml_file_path=f"{task_file}/{id}.yaml"
    # Start a new process
    new_process = run_task(yaml_file_path)
    tasks[id] = {'process': new_process, 'yaml_file': yaml_file_path}
    time.sleep(3)
    if new_process.poll() is not None:
        return jsonify({'restart error': id}), 400
    else:
        return jsonify({'restart success': id}), 202

@app.route('/<id>/delete', methods=['GET'])
def delete_task(id):
    if id not in tasks:
        if os.path.exists(f"{task_file}/{id}.yaml"):
            os.remove(f"{task_file}/{id}.yaml")
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

        return jsonify({'id': id, 'status': 'process be kill and task be deleted'})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=9010)