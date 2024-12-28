import random
import numpy as np
from typing import Tuple, Union, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .distributed import init_distributed
from .logger import LOGGER

# 设置随机种子的函数
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 初始化CUDA环境并返回默认GPU、GPU数量和设备对象的函数
def set_cuda(opts) -> Tuple[bool, int, torch.device]:
    """
    初始化分布式计算的CUDA环境
    """
    if not torch.cuda.is_available():
        assert opts.local_rank == -1, opts.local_rank
        return True, 0, torch.device("cpu")

    # 获取设备设置
    if opts.local_rank != -1:
        init_distributed(opts)
        torch.cuda.set_device(opts.local_rank)
        device = torch.device("cuda", opts.local_rank)
        n_gpu = 1
        default_gpu = dist.get_rank() == 0
        if default_gpu:
            LOGGER.info(f"Found {dist.get_world_size()} GPUs")
    else:
        default_gpu = True
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()

    return default_gpu, n_gpu, device


# 包装模型以支持分布式数据并行处理的函数
def wrap_model(
    model: torch.nn.Module, device: torch.device, local_rank: int
) -> torch.nn.Module:
    model.to(device)

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        # 在DDP包装时，rank0上的参数和缓冲区（即model.state_dict()）会广播到所有其他等级。
    elif torch.cuda.device_count() > 1:
        LOGGER.info("Using data parallel")
        model = torch.nn.DataParallel(model)

    return model


# NoOp类的实现，它在分布式训练中用于无操作
class NoOp(object):
    """ 在分布式训练中用于无操作的工具类 """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return