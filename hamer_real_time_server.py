from hamer_real_time import CameraProcessor
from flask import Flask, jsonify

# Flask Web 服务
app = Flask(__name__)

# 创建处理器实例
processor = CameraProcessor()

@app.route('/start', methods=['GET'])
def start_service():
    """ 启动视频流处理 """
    if not processor.running:
        processor.start()
        print("视频流处理已开始")
        return jsonify({"status": "started", "message": "Video processing started."}), 200
    else:
        return jsonify({"status": "error", "message": "Already running."}), 400

@app.route('/stop', methods=['GET'])
def stop_service():
    """ 停止视频流处理 """
    if processor.running:
        processor.stop()
        print("视频流处理已停止")
        return jsonify({"status": "stopped", "message": "Video processing stopped."}), 200
    else:
        return jsonify({"status": "error", "message": "Not running."}), 400

@app.route('/reset', methods=['GET'])
def reset_init_xyz():
    """ 获取当前处理的结果 """
    flag = processor.set_init_xyz()
    if flag:
        print("初始姿态点已重置！")
    else:
        print("没有检测到右手！")
    return jsonify({"status": "reset", "message": "Reset init xyz."}), 200

@app.route('/results', methods=['GET'])
def get_results():
    """ 获取当前处理的结果 """
    results = processor.get_results()
    if results:
        return jsonify(results), 200
    else:
        return jsonify({"status": "error", "message": "No results available."}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
