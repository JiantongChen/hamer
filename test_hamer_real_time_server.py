import requests
import time


BASE_URL = "http://127.0.0.1:5001"  # Flask 服务的地址

def test_start_service():
    """ 测试启动服务 """
    print("Testing start service...")
    response = requests.get(f"{BASE_URL}/start")
    if response.status_code == 200:
        print("Service started successfully!")
        print(response.json())  # 打印返回的 JSON 响应
    else:
        print("Failed to start service:", response.json())

def test_stop_service():
    """ 测试停止服务 """
    print("Testing stop service...")
    response = requests.get(f"{BASE_URL}/stop")
    if response.status_code == 200:
        print("Service stopped successfully!")
        print(response.json())  # 打印返回的 JSON 响应
    else:
        print("Failed to stop service:", response.json())

def test_reset():
    """ 测试获取结果 """
    print("Testing reset...")
    response = requests.get(f"{BASE_URL}/reset")
    if response.status_code == 200:
        print("Success reset:")
        print(response.json())  # 打印返回的 JSON 响应
    else:
        print("Failed reset:", response.json())

def test_get_results():
    """ 测试获取结果 """
    print("Testing get results...")
    response = requests.get(f"{BASE_URL}/results")
    if response.status_code == 200:
        print("Results received:")
        print(response.json())  # 打印返回的 JSON 响应
    else:
        print("Failed to get results:", response.json())

if __name__ == '__main__':
    # 先启动服务
    test_start_service()

    # 等待一段时间，模拟处理数据
    time.sleep(5)

    test_reset()

    # 获取处理结果
    test_get_results()
    time.sleep(1)
    # 获取处理结果
    test_get_results()
    time.sleep(1)
    # 获取处理结果
    test_get_results()
    time.sleep(1)

    # 停止服务
    test_stop_service()
