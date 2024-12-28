"""
Distributed tools
这段代码提供了一组用于分布式训练的工具函数，包括初始化分布式环境、检查分布式是否可用、收集所有进程的数据、以及减少字典中的值等
"""
import os
from pathlib import Path
from pprint import pformat
import pickle

import torch
import torch.distributed as dist


# 加载分布式初始化参数的函数
def load_init_param(opts):
    """
    加载分布式会合程序的参数

    参数:
        opts: 包含分布式设置的选项对象
    """
    print(opts)
    # 同步文件
    if opts.output_dir != "":
        sync_dir = Path(opts.output_dir).resolve()
        sync_dir.mkdir(parents=True, exist_ok=True)
        sync_file = f"{sync_dir}/.torch_distributed_sync"
    else:
        raise RuntimeError("找不到任何同步目录")

    # 世界大小（world size）
    if opts.world_size != -1:
        world_size = opts.world_size
    elif os.environ.get("WORLD_SIZE", "") != "":
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        raise RuntimeError("找不到任何世界大小")

    # 等级（rank）
    if os.environ.get("RANK", "") != "":
        rank = int(os.environ["RANK"])
    else:
        # 如果没有提供，计算gpu等级
        if opts.node_rank != -1:
            node_rank = opts.node_rank
        elif os.environ.get("NODE_RANK", "") != "":
            node_rank = int(os.environ["NODE_RANK"])
        else:
            raise RuntimeError("找不到任何等级或节点等级")

        if opts.local_rank != -1:
            local_rank = opts.local_rank
        elif os.environ.get("LOCAL_RANK", "") != "":
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            raise RuntimeError("找不到任何等级或本地等级")

        # WARNING: 这假设每个节点有相同数量的GPU
        n_gpus = torch.cuda.device_count()
        rank = local_rank + node_rank * n_gpus
    opts.rank = rank

    return {
        "backend": "nccl",
        "init_method": f"file://{sync_file}",
        "rank": rank,
        "world_size": world_size,
    }


# 初始化分布式环境的函数
def init_distributed(opts):
    init_param = load_init_param(opts)
    rank = init_param["rank"]

    print(f"Init distributed {init_param['rank']} - {init_param['world_size']}")

    dist.init_process_group(**init_param)


# 检查是否是默认GPU的函数
def is_default_gpu(opts) -> bool:
    return opts.local_rank == -1 or dist.get_rank() == 0


# 检查分布式是否可用且已初始化的函数
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# 获取世界大小的函数
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


# 收集所有进程数据的函数
def all_gather(data):
    """
    在任意可pickle数据上运行all_gather（不一定是张量）

    参数:
        data: 任何可pickle的对象

    返回:
        list[data]: 从每个等级收集的数据列表
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # 序列化为张量
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # 获取每个等级的张量大小
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # 从所有等级接收张量
    # 我们填充张量，因为torch all_gather不支持
    # 收集不同形状的张量
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


# 减少字典中值的函数
def reduce_dict(input_dict, average=True):
    """
    将字典中的值从所有进程中减少，以便所有进程都有平均结果。

    参数:
        input_dict (dict): 将减少所有值的字典
        average (bool): 是否执行平均或总和

    返回:
        dict: 经过减少的字典，与input_dict具有相同的字段。
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # 按顺序排序键，以便在进程之间保持一致
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


