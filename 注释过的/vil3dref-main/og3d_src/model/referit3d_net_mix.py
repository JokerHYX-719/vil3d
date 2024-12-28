'''ReferIt3DNetMix的神经网络模型，它是一个“混合”模型，结合了两个不同配置的ReferIt3DNet模型——一个教师模型和一个学生模型。'''
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from transformers import BertConfig, BertModel

from .obj_encoder import GTObjEncoder, PcdObjEncoder
from .mmt_module import MMT
from .cmt_module import CMT
from .referit3d_net import get_mlp_head, freeze_bn
from .referit3d_net import ReferIt3DNet


# 定义混合模型类
class ReferIt3DNetMix(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # 教师评估模式的配置
        self.teacher_eval_mode = config.get('teacher_eval_mode', False)

        # 创建教师模型配置，并初始化教师模型
        teacher_model_cfg = copy.deepcopy(config)
        teacher_model_cfg.model_type = 'gtlabel'
        teacher_model_cfg.obj_encoder.use_color_enc = teacher_model_cfg.obj_encoder.teacher_use_color_enc
        self.teacher_model = ReferIt3DNet(teacher_model_cfg, device)

        # 创建学生模型配置，并初始化学生模型
        student_model_cfg = copy.deepcopy(config)
        student_model_cfg.model_type = 'gtpcd'
        student_model_cfg.obj_encoder.use_color_enc = student_model_cfg.obj_encoder.student_use_color_enc
        self.student_model = ReferIt3DNet(student_model_cfg, device)

        # 冻结教师模型的参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    # 准备批次数据，将其移动到指定的设备上
    def prepare_batch(self, batch):
        outs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                outs[key] = value.to(self.device)
            else:
                outs[key] = value
        return outs

    # 前向传播方法
    def forward(self, batch: dict, compute_loss=False, is_test=False) -> dict:
        batch = self.prepare_batch(batch)

        # 如果处于教师评估模式，则将教师模型设置为评估模式
        if self.teacher_eval_mode:
            self.teacher_model.eval()

        # 准备教师模型的输入数据
        batch_teacher = {
            'obj_fts': batch['obj_gt_fts'],
            'obj_colors': batch['obj_colors'],
            'obj_locs': batch['obj_locs'],
            'obj_masks': batch['obj_masks'],
            'txt_ids': batch['txt_ids'],
            'txt_masks': batch['txt_masks']
        }
        teacher_outs = self.teacher_model(
            batch_teacher, compute_loss=False,
            output_attentions=True, output_hidden_states=True,
        )
        # 将教师模型的输出结果进行detach，使其在反向传播时不会计算梯度
        for k, v in teacher_outs.items():
            if isinstance(v, list):
                teacher_outs[k] = [x.detach() for x in v]
            else:
                teacher_outs[k] = v.detach()

        # 使用学生模型进行前向传播
        student_outs = self.student_model(
            batch, compute_loss=False,
            output_attentions=True, output_hidden_states=True,
        )

        # 如果需要计算损失，则调用计算损失的方法
        if compute_loss:
            losses = self.compute_loss(teacher_outs, student_outs, batch)
            return student_outs, losses

        return student_outs

    # 计算损失函数，包括蒸馏损失
    def compute_loss(self, teacher_outs, student_outs, batch):
        losses = self.student_model.compute_loss(student_outs, batch)

        # 如果配置了蒸馏损失，则计算蒸馏损失
        if self.config.losses.distill_cross_attns > 0:
            # 计算交叉注意力的蒸馏损失
            cross_attn_masks = batch['obj_masks'].unsqueeze(2) * batch['txt_masks'].unsqueeze(1)
            cross_attn_masks = cross_attn_masks.float()
            cross_attn_sum = cross_attn_masks.sum()
            for i in range(self.config.mm_encoder.num_layers):
                mse_loss = (teacher_outs['all_cross_attns'][i] - student_outs['all_cross_attns'][i]) ** 2
                mse_loss = torch.sum(mse_loss * cross_attn_masks) / cross_attn_sum
                losses['cross_attn_%d' % i] = mse_loss * self.config.losses.distill_cross_attns
                losses['total'] += losses['cross_attn_%d' % i]

        if self.config.losses.distill_self_attns > 0:
            # 计算自注意力的蒸馏损失
            self_attn_masks = batch['obj_masks'].unsqueeze(2) * batch['obj_masks'].unsqueeze(1)
            self_attn_masks = self_attn_masks.float()
            self_attn_sum = self_attn_masks.sum()
            for i in range(self.config.mm_encoder.num_layers):
                mse_loss = (teacher_outs['all_self_attns'][i] - student_outs['all_self_attns'][i]) ** 2
                mse_loss = torch.sum(mse_loss * self_attn_masks) / self_attn_sum
                losses['self_attn_%d' % i] = mse_loss * self.config.losses.distill_self_attns
                losses['total'] += losses['self_attn_%d' % i]

        if self.config.losses.distill_hiddens > 0:
            # 计算隐藏状态的蒸馏损失
            hidden_masks = batch['obj_masks'].unsqueeze(2).float()
            hidden_sum = hidden_masks.sum() * self.config.hidden_size
            for i in range(self.config.mm_encoder.num_layers + 1):
                mse_loss = (teacher_outs['all_hidden_states'][i] - student_outs['all_hidden_states'][i]) ** 2
                mse_loss = torch.sum(mse_loss * hidden_masks) / hidden_sum
                losses['hidden_state_%d' % i] = mse_loss * self.config.losses.distill_hiddens
                losses['total'] += losses['hidden_state_%d' % i]

        return losses
