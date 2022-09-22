import pynvml
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        help="name")
    parser.add_argument(
        "--cuda",
        help="cuda device",
        nargs='+', type=int)
    
    args = parser.parse_args()
    return args

args = parse_args()
pynvml.nvmlInit()
handles = []
log_file_names = []
for device_id in args.cuda:
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    handles.append(handle)
    log_file_name = "{}_{}.csv".format(args.name, device_id)
    log_file_names.append(log_file_name)

while True:
    for i in range(len(handles)):
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handles[i])
        use_mem = meminfo.used //(1024 * 1024)  
        with open(log_file_names[i], 'a+') as f:
            f.write("{}".format(use_mem) + '\n')
    time.sleep(1)