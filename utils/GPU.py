import os, tensorflow as tf
# Hide GPU from visible devices
def InitializeGPU():
    """
    2024-12-17 12:39:33.030218: I external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:1193] failed to allocate 2.2KiB (2304 bytes) from device: RESOURCE_EXHAUSTED: : CUDA_ERROR_OUT_OF_MEMORY: out of memory
    https://stackoverflow.com/questions/39465503/cuda-error-out-of-memory-in-tensorflow
    https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    https://www.tensorflow.org/guide/gpu
    """
    #tf.config.set_visible_devices([], 'GPU')
    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"{len(gpus)} GPUs available")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"

def SetMemoryLimit(size: int):
    gpus = tf.config.list_physical_devices('GPU')
    print(f"{len(gpus)} GPUs available")
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=size)])

def UseCPU():
    # Get a list of all physical devices
    physical_devices = tf.config.list_physical_devices()

    # Filter for CPU devices
    cpu_devices = [device for device in physical_devices if device.device_type == 'CPU']

    # Set only CPU devices as visible
    tf.config.set_visible_devices(cpu_devices)    