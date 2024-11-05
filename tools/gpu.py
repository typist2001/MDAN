#简单使用
# pip install nvidia-ml-py

from pynvml import *
nvmlInit()     #初始化
print("Driver: ",nvmlSystemGetDriverVersion())  #显示驱动信息
#>>> Driver: 384.xxx
def convert_bytes(size):
    # 千字节（KB）
    KB = 1024
    # 兆字节（MB）
    MB = KB * 1024
    # 千兆字节（GB）
    GB = MB * 1024

    if size < KB:
        return f"{size} B"
    elif size < MB:
        return f"{size / KB:.2f} KB"
    elif size < GB:
        return f"{size / MB:.2f} MB"
    else:
        return f"{size / GB:.2f} GB"


#查看设备
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("GPU", i, ":", nvmlDeviceGetName(handle))
#>>>
#GPU 0 : b'GeForce GTX 1080 Ti'
#GPU 1 : b'GeForce GTX 1080 Ti'

#查看显存、温度、风扇、电源
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Memory Total: ",convert_bytes(info.total))
print("Memory Free: ",convert_bytes(info.free))
print("Memory Used: ",convert_bytes(info.used))

print("Temperature is %d C"%nvmlDeviceGetTemperature(handle,0))
print("Power ststus",nvmlDeviceGetPowerState(handle))


#最后要关闭管理工具
nvmlShutdown()


#nvmlDeviceXXX有一系列函数可以调用，包括了NVML的大多数函数。
#具体可以参考：https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries
