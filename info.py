import psutil
import platform
from datetime import datetime

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

print("-"*20, "System Information ", "-"*20)
uname = platform.uname()
print(f"System   : {uname.system}")
print(f"Node Name: {uname.node}")
print(f"Release  : {uname.release}")
print(f"Version  : {uname.version}")
print(f"Machine  : {uname.machine}")
print(f"Processor: {uname.processor}")

print ()
print("-"*20, "CPU Info           ", "-"*20)
# number of cores
print ("Physical cores    :", psutil.cpu_count(logical=False))
print ("Total cores       :", psutil.cpu_count(logical=True))
# CPU frequencies
cpufreq = psutil.cpu_freq()
print(f"Max Frequency     : {cpufreq.max:.2f}Mhz")
print(f"Min Frequency     : {cpufreq.min:.2f}Mhz")
print(f"Current Frequency : {cpufreq.current:.2f}Mhz")
# CPU usage
print ("CPU Usage Per Core:")
print('--------------------------')
for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
    print(f" - Core {i}: {percentage}%")
print('--------------------------')
print(f"Total CPU Usage  : {psutil.cpu_percent()}%")

print ()
print("-"*20, "Memory Information ", "-"*20)
# get the memory details
svmem = psutil.virtual_memory()
print(f"Total     : {get_size(svmem.total)}")
print(f"Available : {get_size(svmem.available)}")
print(f"Used      : {get_size(svmem.used)}")
print(f"Percentage: {svmem.percent}%")
print("-"*20, "SWAP               ", "-"*20)
# get the swap memory details (if exists)
swap = psutil.swap_memory()
print(f"Total     : {get_size(swap.total)}")
print(f"Free      : {get_size(swap.free)}")
print(f"Used      : {get_size(swap.used)}")
print(f"Percentage: {swap.percent}%")

print ()
# Disk Information
print("-"*20, "Disk Information   ", "-"*20)
print("Partitions and Usage:")
# get all disk partitions
partitions = psutil.disk_partitions()
for partition in partitions:
    print(f"* Device: {partition.device}")
    print(f"  Mountpoint: {partition.mountpoint}")
    print(f"  File system type: {partition.fstype}")
    try:
        partition_usage = psutil.disk_usage(partition.mountpoint)
    except PermissionError:
        # this can be catched due to the disk that
        # isn't ready
        continue
    print(f"  Total Size: {get_size(partition_usage.total)}")
    print(f"  Used      : {get_size(partition_usage.used)}")
    print(f"  Free      : {get_size(partition_usage.free)}")
    print(f"  Percentage: {partition_usage.percent}%")
# get IO statistics since boot
print('--------------------------')
disk_io = psutil.disk_io_counters()
print(f"Total read  : {get_size(disk_io.read_bytes)}")
print(f"Total write : {get_size(disk_io.write_bytes)}")

print ()
# Network information
              #1234567890123456789
print("-"*20, "Network Information", "-"*20)
# get all network interfaces (virtual and physical)
if_addrs = psutil.net_if_addrs()
for interface_name, interface_addresses in if_addrs.items():
    for address in interface_addresses:
        print(f"* Interface: {interface_name}")
        if str(address.family) == 'AddressFamily.AF_INET':
            print(f"  IP Address   : {address.address}")
            print(f"  Netmask      : {address.netmask}")
            print(f"  Broadcast IP : {address.broadcast}")
        elif str(address.family) == 'AddressFamily.AF_PACKET':
            print(f"  MAC Address  : {address.address}")
            print(f"  Netmask      : {address.netmask}")
            print(f"  Broadcast MAC: {address.broadcast}")
# get IO statistics since boot
print('--------------------------')
net_io = psutil.net_io_counters()
print(f"Total Bytes Sent    : {get_size(net_io.bytes_sent)}")
print(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")

print ()
# GPU information
import GPUtil
from tabulate import tabulate
print("-"*62, "GPU Details", "-"*62)
gpus = GPUtil.getGPUs()
list_gpus = []
for gpu in gpus:
    # get the GPU id
    gpu_id = gpu.id
    # name of GPU
    gpu_name = gpu.name
    # get % percentage of GPU usage of that GPU
    gpu_load = f"{gpu.load*100}%"
    # get free memory in MB format
    gpu_free_memory = f"{gpu.memoryFree}MB"
    # get used memory
    gpu_used_memory = f"{gpu.memoryUsed}MB"
    # get total memory
    gpu_total_memory = f"{gpu.memoryTotal}MB"
    # get GPU temperature in Celsius
    gpu_temperature = f"{gpu.temperature} °C"
    gpu_uuid = gpu.uuid
    list_gpus.append((
        gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
        gpu_total_memory, gpu_temperature, gpu_uuid
    ))

print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory", "temperature", "uuid")))
print ()
print ()
