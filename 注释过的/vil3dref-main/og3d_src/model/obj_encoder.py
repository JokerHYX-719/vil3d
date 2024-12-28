import copy

import torch
import torch.nn as nn
import einops

from .backbone.point_net_pp import PointNetPP

class GTObjEncoder(nn.Module):
    '''用于编码对象的特征向量。它可以根据配置参数决定是使用one-hot编码还是直接使用原始特征向量。One-hot编码是一种将分类数据转换为二进制向量的编码方式'''
    def __init__(self, config, hidden_size):
        super().__init__()
        # 复制配置参数并设置隐藏层大小
        self.config = copy.deepcopy(config)
        self.config.hidden_size = hidden_size

        # 如果使用one-hot编码，则使用嵌入层；否则，使用线性层
        if self.config.onehot_ft:
            self.ft_linear = [nn.Embedding(self.config.num_obj_classes, self.config.hidden_size)]
        else:
            self.ft_linear = [nn.Linear(self.config.dim_ft, self.config.hidden_size)]
        # 添加层归一化和序列模块
        self.ft_linear.append(nn.LayerNorm(self.config.hidden_size))
        self.ft_linear = nn.Sequential(*self.ft_linear)

        # Dropout层
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, obj_fts):
        # 编码对象特征
        obj_embeds = self.ft_linear(obj_fts)
        # 应用dropout
        obj_embeds = self.dropout(obj_embeds)
        return obj_embeds

class PcdObjEncoder(nn.Module):
    '''用于编码点云数据。它使用PointNet++网络来提取点云的特征'''
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 初始化PointNet++网络
        self.pcd_net = PointNetPP(
            sa_n_points=config.sa_n_points,
            sa_n_samples=config.sa_n_samples,
            sa_radii=config.sa_radii,
            sa_mlps=config.sa_mlps,
        )
        # Dropout层
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, obj_pcds):
        # 获取批次大小和对象数量
        batch_size, num_objs, _, _ = obj_pcds.size()
        # 使用PointNet++网络处理每个点云数据
        obj_embeds = []
        for i in range(batch_size):
            obj_embeds.append(self.pcd_net(obj_pcds[i]))
        obj_embeds = torch.stack(obj_embeds, 0)

        # 应用dropout
        obj_embeds = self.dropout(obj_embeds)
        return obj_embeds


class ObjColorEncoder(nn.Module):
    '''用于编码对象的颜色信息。它将颜色的高斯混合模型（GMM）参数通过一个线性层和ReLU激活函数进行编码。'''
    def __init__(self, hidden_size, dropout=0):
        super().__init__()
        # 初始化线性层、ReLU激活函数、层归一化和Dropout
        self.ft_linear = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout)
        )

    def forward(self, obj_colors):
        # obj_colors的形状为(batch, nobjs, 3, 4)，其中前3维是颜色值，后4维是GMM参数
        gmm_weights = obj_colors[..., :1]
        gmm_means = obj_colors[..., 1:]

        # 将GMM的均值通过线性层编码，然后与权重相乘并求和
        embeds = torch.sum(self.ft_linear(gmm_means) * gmm_weights, 2)
        return embeds
        