import os
import time
import gpustat
import psutil
import yaml


def read_yaml(file):
    with open(file) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        return data


def wait_for_processes(pids, timeout_seconds=60):
    if pids is not None:
        attempts = 1
        while sum([psutil.pid_exists(pid) for pid in pids]) > 0:
            print(f'(#{attempts}) at least one process from {pids} is still running, waiting {timeout_seconds} seconds...')
            attempts += 1
            time.sleep(timeout_seconds)


def wait_for_gpus_of_user(gpus, timeout_seconds=60):
    """
    This method waits `timeout_seconds` for all processes of current user to finish on all GPU cards with IDs in `gpus`
    """
    attempts = 1
    user = os.getlogin()
    while True:
        gpus_stat = gpustat.new_query().gpus # get the status of GPUs
        processes_used_by_user = 0
        for i in gpus: # i is the ID of a GPU in CUDA_VISIBLE_DEVICES
            for proc in gpus_stat[i].processes: # p is a dict containing keys (username, command, gpu_memory_usage, pid)
                if proc['username'] == user:
                    processes_used_by_user += 1
        if processes_used_by_user == 0: # the script can run
            return

        # block the script here
        print(f'(#{attempts}) {user} has processes running on at least one GPU from {gpus}, waiting {timeout_seconds} seconds...')
        attempts += 1
        time.sleep(timeout_seconds)