import os
import time
import psutil
import yaml
import platform
import random


def on_windows():
    return platform.system().lower() == 'windows'


if not on_windows():
    import gpustat


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


def wait_for_gpus_of_user(gpus, max_jobs=None, timeout_seconds=60):
    """
    This method waits `timeout_seconds` for all processes of current user to finish on all GPU cards with IDs in `gpus`
    """
    attempts = 1
    user = os.getlogin()
    while True:
        gpus_stat = gpustat.new_query().gpus # get the status of GPUs
        processes_used_by_user = 0
        for i in gpus: # i is the ID of a GPU in CUDA_VISIBLE_DEVICES
            for proc in gpus_stat[i].processes: # proc is a dict containing keys (username, command, gpu_memory_usage, pid)
                if proc['username'] == user:
                    processes_used_by_user += 1
        if max_jobs is None: # run when GPUs are free
            if processes_used_by_user == 0: # the script can run now
                return
        else: # run when there are less than max_jobs
            if processes_used_by_user < max_jobs:
                return

        # block the script here
        print(f'(#{attempts}) {user} has processes running on at least one GPU from {gpus}, waiting {timeout_seconds} seconds...')
        attempts += 1
        time.sleep(timeout_seconds)


def get_free_gpu(gpus, max_jobs, attempts=0):
    """
    Returns the first GPU from `gpus` that has less than `max_jobs` running for the current user
    """
    user = os.getlogin()
    can_run_on_gpu = [False] * len(gpus) # flags telling whether we can run the script on a gpu in `gpus`
    gpu_proc_count = [0] * len(gpus)

    gpu_stat = gpustat.new_query().gpus
    for i, gpu_id in enumerate(gpus):
        user_processes = [p for p in gpu_stat[gpu_id].processes if p['username'] == user]
        gpu_proc_count[i] = len(user_processes)
        # can_run_on_gpu[i] = (len(user_processes) < max_jobs)

    least_busy_gpu_count = None
    least_busy_gpu_index = None
    for i, count in enumerate(gpu_proc_count):
        if count < max_jobs:
            if least_busy_gpu_count is None or count < least_busy_gpu_count:
                least_busy_gpu_count = count
                least_busy_gpu_index = gpus[i]

    if least_busy_gpu_count is not None:
        return least_busy_gpu_index

    # # if one flag is True, then pick the gpu from that index and run on it
    # available_gpus = [i for i, flag in enumerate(can_run_on_gpu) if flag]
    # if len(available_gpus) > 0:
    #     gpu = random.choice(available_gpus)
    #     return gpu

    # wait 60 seconds then try again
    print(f'All GPUs in {gpus} have {max_jobs} jobs, waiting 60 seconds...')
    time.sleep(60)
    get_free_gpu(gpus, max_jobs, attempts + 1)
