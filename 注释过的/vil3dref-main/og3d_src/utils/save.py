"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import json
import os
import torch


# 保存训练配置的函数
def save_training_meta(args):
    # 创建日志和检查点目录
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ckpts'), exist_ok=True)

    # 将配置参数保存为JSON文件
    with open(os.path.join(args.output_dir, 'logs', 'config.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)


# 模型保存器类的实现
class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_epoch', suffix='pt'):
        """
        初始化模型保存器。

        参数:
            output_dir (str): 输出目录路径。
            prefix (str): 模型文件名前缀。
            suffix (str): 模型文件名后缀。
        """
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, epoch, optimizer=None, save_latest_optim=False):
        """
        保存模型和优化器状态。

        参数:
            model (torch.nn.Module): 要保存的模型。
            epoch (int): 当前训练周期。
            optimizer (torch.optim.Optimizer, optional): 优化器。如果提供，其状态将被保存。
            save_latest_optim (bool): 是否保存最新的优化器状态。
        """
        # 构造输出模型文件名
        output_model_file = os.path.join(self.output_dir,
                                 f"{self.prefix}_{epoch}.{self.suffix}")
        # 创建模型状态字典
        state_dict = {}
        for k, v in model.state_dict().items():
            if k.startswith('module.'):
                k = k[7:]  # 移除分布式训练的前缀
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.cpu()  # 将模型参数转移到CPU并保存
            else:
                state_dict[k] = v
        # 保存模型状态
        torch.save(state_dict, output_model_file)
        # 如果提供了优化器，保存其状态
        if optimizer is not None:
            dump = {'epoch': epoch, 'optimizer': optimizer.state_dict()}
            # 如果优化器有_amp_stash属性，则跳过（用于混合精度训练）
            if hasattr(optimizer, '_amp_stash'):
                pass
            if save_latest_optim:
                # 保存最新的优化器状态
                torch.save(dump, f'{self.output_dir}/train_state_lastest.pt')
            else:
                # 保存当前周期的优化器状态
                torch.save(dump, f'{self.output_dir}/train_state_{epoch}.pt')
        return output_model_file

    def remove_previous_models(self, cur_epoch):
        """
        删除之前的模型文件。

        参数:
            cur_epoch (int): 当前周期。
        """
        # 遍历输出目录中的所有文件
        for saved_model_name in os.listdir(self.output_dir):
            if saved_model_name.startswith(self.prefix):
                # 提取保存的模型周期
                saved_model_epoch = int(os.path.splitext(saved_model_name)[0].split('_')[-1])
                # 如果保存的模型周期不是当前周期，则删除该模型文件
                if saved_model_epoch != cur_epoch:
                    os.remove(os.path.join(self.output_dir, saved_model_name))

