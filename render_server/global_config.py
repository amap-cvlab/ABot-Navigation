import threading

# 全局CUDA初始化锁
cuda_init_lock = threading.Lock()