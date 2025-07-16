import time
import threading
from collections import deque

BUFFER_TIME = 1.0  # 0.5s 缓存
short_term_data = {}  # 存储短时数据
single_value_data = {}  # 存储单个数值
prediction_data = {}  # 预测/规划数据
lock = threading.Lock()

def store_short_term(name, coords):
    """存储短时数据"""
    with lock:
        if name not in short_term_data:
            short_term_data[name] = deque()
        short_term_data[name].append((time.time(), coords))

def get_recent_short_term(name):
    """获取最近 0.5s 的短时数据"""
    with lock:
        if name in short_term_data:
            current_time = time.time()
            short_term_data[name] = deque(filter(lambda x: x[0] >= current_time - BUFFER_TIME, short_term_data[name]))
            return list(short_term_data[name])
    return []

def store_single_value(name, value):
    """存储单个数据（如当前 TCP 位置、当前距离）"""
    with lock:
        single_value_data[name] = value

def get_single_value(name):
    """获取单个存储数据"""
    with lock:
        return single_value_data.get(name, None)

def store_prediction(name, data):
    """存储预测或规划数据"""
    with lock:
        prediction_data[name] = data

def get_prediction(name):
    """获取存储的预测或规划数据"""
    with lock:
        return prediction_data.get(name, None)

def buffer_cleaner():
    """定期清理短时数据"""
    while True:
        time.sleep(0.1)
        with lock:
            for name in short_term_data:
                current_time = time.time()
                short_term_data[name] = deque(filter(lambda x: x[0] >= current_time - BUFFER_TIME, short_term_data[name]))
