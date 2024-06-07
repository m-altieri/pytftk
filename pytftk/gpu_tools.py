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
    Note: calling use_devices() before this function will prevent this
    function from seeing all physical GPUs, not allowing it to find the true
    most free one. Call this function before use_devices() to ensure to find
    the most free GPU.

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

    Args:
        device_index (int or list of ints): the index of a single device or a
        list of indexes of multiple devices. If -1, use all devices.

    Raises:
        IndexError

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


def await_avail_memory(device_idx, min_bytes, interval=60):
    """Pause the execution synchronously until enough GPU memory appears to be
    free for the provided GPU indexes.

    Args:
        device_idx (int or list of ints): index(es) of the GPU to wait for.
        min_bytes (int): minimum amount of free memory (in bytes) that all
        devices in device_idx need to have in order to resume execution.
        interval (int, optional): amount of seconds to wait for in between
        available memory checks. Defaults to 60.
    """

    # if device_index is an int, make it a list
    if isinstance(device_idx, int):
        device_idx = [device_idx]

    # if device_index is -1, use all gpus
    if device_idx == [-1]:
        device_idx = list(range(len(tf.config.list_physical_devices("GPU"))))

    for d in device_idx:
        avail = get_avail_memory(d)
        while avail < min_bytes:
            print(
                f"{Style.BRIGHT}{Fore.YELLOW}Device {d} has "
                + f"{avail / GiB :.2f} GiB left, but at least "
                + f"{min_bytes / GiB:.2f} are needed to start. Waiting "
                + f"{interval} seconds to see if it frees up...{Style.RESET_ALL}",
            )
            time.sleep(interval)
            avail = get_avail_memory(d)
