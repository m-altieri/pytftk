import time
import pynvml
import tensorflow as tf
from colorama import Fore, Style


def use_devices(device_index):
    """Set GPU devices to use and enable memory growth on them.

    Args:
        device_index (int or list of ints): the index of a single device or a
        list of indexes of multiple devices.
    """

    # if device_index is an int, make it a list
    if isinstance(device_index, int):
        device_index = [device_index]

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


def get_avail_memory(device_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.free


def await_avail_memory(device_index, min_bytes, interval=60):
    avail = get_avail_memory(device_index)
    while avail < min_bytes:
        print(
            f"{Style.BRIGHT}{Fore.YELLOW}Device {device_index} has "
            + f"{avail / 1024**3 :.2f} GiB left, but at least "
            + f"{min_bytes / 1024**3:.2f} are needed to start. Waiting "
            + f"{interval} seconds to see if it frees up...{Style.RESET_ALL}",
        )
        time.sleep(interval)
        avail = get_avail_memory(device_index)
