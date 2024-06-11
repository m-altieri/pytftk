import time
import pynvml
import tensorflow as tf
from colorama import Fore, Style


KiB = 1024
MiB = 1024**2
GiB = 1024**3


def get_freeest_gpu():
    """
    Get the index and the free memory of the GPU with the most free memory.

    Returns:
        tuple (int, int): the index and the free memory in bytes of the GPU
        with the most free memory.
    """

    visible_devices = tf.config.get_visible_devices("GPU")

    freeest = (0, 0)
    for current_idx in range(len(visible_devices)):
        _, freeest_mem = freeest
        current_mem = get_avail_memory(current_idx)
        if current_mem > freeest_mem:
            freeest = current_idx, current_mem

    print(
        f"{Fore.CYAN}[Experimental] Automatic detection found GPU {freeest[0]} "
        + f"to be the most free with {freeest[1] / GiB :.2f} GiB.{Fore.RESET}"
    )
    return freeest


def use_devices(device_index):
    """Set GPU devices to use and enable memory growth on them.
    Note: calling use_devices() will prevent other functions from seeing all 
    physical GPUs, not allowing them to find e.g. the true most free one. 
    Call use_devices() after you are sure you have selected the GPU(s) to use
    (e.g. after finding the most free one).

    Args:
        device_index (int or list of ints): the index of a single device or a
        list of indexes of multiple devices. If -1, use all devices.

    Raises:
        IndexError: if the provided device_index is out of range.

    Returns:
        list: For convenience, return the list of the correctly set visible
        devices.
    """

    # if device_index is an int, make it a list
    if isinstance(device_index, int):
        device_index = [device_index]

    # if device_index is -1, use all gpus
    if device_index == [-1]:
        device_index = list(range(len(tf.config.list_physical_devices("GPU"))))

    # set only specified devices as visible
    try:
        tf.config.set_visible_devices(
            [tf.config.list_physical_devices("GPU")[d] for d in device_index], "GPU"
        )
    except IndexError as e:
        print(
            f"[ERROR] Can't use device {device_index}: index is out of range. \
            The physical GPU devices appear to be \
            {tf.config.list_physical_devices('GPU')}."
        )
        raise e
    assert len(tf.config.get_visible_devices("GPU")) == len(device_index)

    # enable memory growth on all visible devices
    visible_devices = tf.config.get_visible_devices("GPU")
    for device in visible_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f"Using and enabling memory growth on device {device}.")

    return visible_devices


def get_avail_memory(device_index):
    """Get the available memory of a GPU device.

    Args:
        device_index (int): the index of the GPU device.

    Returns:
        int: the available memory in bytes.
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.free


def await_avail_memories(device_idxs, min_bytes, interval=60):
    """Pause the execution synchronously until all GPUs in device_idxs have 
    enough free memory.

    Args:
        device_idxs (list): indexes of the GPUs to wait for. Cannot contain -1.
        min_bytes (int): minimum amount of free memory (in bytes) that the
        device with index device_idx need to have in order to resume execution.
        interval (int, optional): amount of seconds to wait for in between
        available memory checks. Defaults to 60.

    Raises:
        ValueError: if device_idxs is not a list.
        ValueError: if device_idxs contains a -1.
    """    
    if not isinstance(device_idxs, list):
        raise ValueError(
            f"device_idxs ({device_idxs}) must be list, found {type(device_idxs)} instead."
        )
    if -1 in device_idxs:
        raise ValueError(
            f"device_idxs ({device_idxs}) cannot contain -1, found {device_idxs} instead."
        )
    
    available = True
    while True:
        for d in device_idxs:
            _, instantly_available = await_avail_memory(d, min_bytes, interval)
            available &= instantly_available
        if available:
            break

def await_avail_memory(device_idx, min_bytes, interval=60): 
    """Pause the execution synchronously until enough GPU memory appears to be
    free for the provided GPU index.

    Args:
        device_idx (int): index of the GPU to wait for. If -1, wait until any
        GPU is available.
        min_bytes (int): minimum amount of free memory (in bytes) that the
        device with index device_idx need to have in order to resume execution.
        interval (int, optional): amount of seconds to wait for in between
        available memory checks. Defaults to 60.

    Raises:
        ValueError: if device_idx is not an int.

    Returns:
        int: the index of the first GPU index, among the ones in device_idx,
        whose available memory becomes higher than min_bytes.
        bool: True if the memory was instantly available, False if execution
        was paused to wait.
    """

    # if device_index is an int, make it a list
    if not isinstance(device_idx, int):
        raise ValueError(
            f"device_idx ({device_idx}) must be int, found {type(device_idx)} instead."
        )
    device_idx = [device_idx]

    # if device_index is -1, use all gpus
    if device_idx == [-1]:
        device_idx = list(range(len(tf.config.list_physical_devices("GPU"))))

    instantly_available = True
    avail = [get_avail_memory(d) for d in device_idx]
    while all([m < min_bytes for m in avail]):
        print(
            f"{Style.BRIGHT}{Fore.YELLOW}[WARN] Device(s) {device_idx} have "
            + f"{[f"{a / GiB :2f}" for a in avail]} GiB left, but at least "
            + f"{min_bytes / GiB:.2f} are needed to start. Waiting "
            + f"{interval} seconds to see if they free up...{Style.RESET_ALL}",
        )
        time.sleep(interval)
        instantly_available = False
        avail = [get_avail_memory(d) for d in device_idx]
    
    return device_idx[avail.index(max(avail))], instantly_available
