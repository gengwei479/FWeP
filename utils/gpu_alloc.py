from pynvml import *
def use_gpu(used_percentage=0.75):
    '''
    不使用显存占用率高于used_percentage的gpu
    :param used_percentage:
    :return:
    '''
    nvmlInit()
    gpu_num = nvmlDeviceGetCount()
    out = ""
    for i in range(gpu_num):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_percentage_real = info.used / info.total
        if out == "":
            if used_percentage_real < used_percentage:
                out += str(i)
        else:
            if used_percentage_real < used_percentage:
                out += "," + str(i)
    nvmlShutdown()
    return out


def use_auto_gpus(LlmClassType, config, checkpoint, cuda_list, memory = '80GiB'):
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
    with init_empty_weights():
        model = LlmClassType(config)
    # max_memory = {int(cuda): memory for cuda in cuda_list}
    # device_map = infer_auto_device_map(model, max_memory = max_memory, 
    #                                    no_split_module_classes = LlmClassType._no_split_modules)
    # print(device_map)
    model = load_checkpoint_and_dispatch(model, checkpoint, device_map = 'auto', #'auto', 
                                            no_split_module_classes = LlmClassType._no_split_modules)
    return model
