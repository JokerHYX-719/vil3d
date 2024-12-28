'''定义了一个名为ReferIt3DNet的神经网络模型，它结合了多种编码器来处理图像、文本和3D对象信息'''
import math
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from transformers import BertConfig, BertModel

from .obj_encoder import GTObjEncoder, PcdObjEncoder, ObjColorEncoder
from .txt_encoder import GloveGRUEncoder
from .mmt_module import MMT
from .cmt_module import CMT


# 定义一个多层感知机（MLP）头部，用于分类或回归任务
def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size // 2),
        nn.ReLU(),
        nn.LayerNorm(hidden_size // 2, eps=1e-12),
        nn.Dropout(dropout),
        nn.Linear(hidden_size // 2, output_size)
    )
    # 输出(batch_size, num_objects, 1)


# 冻结BatchNorm层的参数
def freeze_bn(m):
    '''Freeze BatchNorm Layers'''
    for layer in m.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


class ReferIt3DNet(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # 根据模型类型初始化对象编码器
        config.obj_encoder.num_obj_classes = config.num_obj_classes
        if self.config.model_type == 'gtlabel':
            self.obj_encoder = GTObjEncoder(config.obj_encoder, config.hidden_size)
        elif self.config.model_type == 'gtpcd':
            self.obj_encoder = PcdObjEncoder(config.obj_encoder)
        # 如果配置中指定冻结对象编码器的BatchNorm层或参数，则执行冻结操作
        if self.config.obj_encoder.freeze:
            freeze_bn(self.obj_encoder)
            for p in self.obj_encoder.parameters():
                p.requires_grad = False
        if self.config.obj_encoder.freeze_bn:
            freeze_bn(self.obj_encoder)

        # 如果使用颜色编码器，则初始化它
        if self.config.obj_encoder.use_color_enc:
            self.obj_color_encoder = ObjColorEncoder(config.hidden_size, config.obj_encoder.dropout)

        # 根据配置初始化文本编码器，可以是GRU或BERT模型
        if self.config.txt_encoder.type == 'gru':
            self.txt_encoder = GloveGRUEncoder(config.hidden_size, config.txt_encoder.num_layers)
        else:
            txt_bert_config = BertConfig(
                hidden_size=config.hidden_size,
                num_hidden_layers=config.txt_encoder.num_layers,
                num_attention_heads=12, type_vocab_size=2
            )
            self.txt_encoder = BertModel.from_pretrained(
                'bert-base-uncased', config=txt_bert_config
            )
        # 如果配置中指定冻结文本编码器的参数，则执行冻结操作
        if self.config.txt_encoder.freeze:
            for p in self.txt_encoder.parameters():
                p.requires_grad = False

        # 根据配置初始化多模态编码器，可以是MMT或CMT模型
        mm_config = EasyDict(config.mm_encoder)
        mm_config.hidden_size = config.hidden_size
        mm_config.num_attention_heads = 12
        mm_config.dim_loc = config.obj_encoder.dim_loc
        if self.config.mm_encoder.type == 'cmt':
            self.mm_encoder = CMT(mm_config)
        elif self.config.mm_encoder.type == 'mmt':
            self.mm_encoder = MMT(mm_config)

        # 初始化目标3D头部（用于分类或回归任务）
        self.og3d_head = get_mlp_head(
            config.hidden_size, config.hidden_size,
            1, dropout=config.dropout
        )

        # 如果配置中指定了损失函数的权重，则初始化相应的头部
        if self.config.losses.obj3d_clf > 0:
            self.obj3d_clf_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                config.num_obj_classes, dropout=config.dropout
            )
        if self.config.losses.obj3d_clf_pre > 0:
            self.obj3d_clf_pre_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                config.num_obj_classes, dropout=config.dropout
            )
            if self.config.obj_encoder.freeze:
                for p in self.obj3d_clf_pre_head.parameters():
                    p.requires_grad = False
        if self.config.losses.obj3d_reg > 0:
            self.obj3d_reg_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                3, dropout=config.dropout
            )
        if self.config.losses.txt_clf > 0:
            self.txt_clf_head = get_mlp_head(
                config.hidden_size, config.hidden_size,
                config.num_obj_classes, dropout=config.dropout
            )

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

    def forward(
            self, batch: dict, compute_loss=False, is_test=False,
            output_attentions=False, output_hidden_states=False,
    ) -> dict:
        batch = self.prepare_batch(batch)

        # 如果冻结了对象编码器或其BatchNorm层，则在前向传播中使用detached的变量
        if self.config.obj_encoder.freeze or self.config.obj_encoder.freeze_bn:
            freeze_bn(self.obj_encoder)
        obj_embeds = self.obj_encoder(batch['obj_fts'])
        if self.config.obj_encoder.freeze:
            obj_embeds = obj_embeds.detach()
        if self.config.obj_encoder.use_color_enc:
            obj_embeds = obj_embeds + self.obj_color_encoder(batch['obj_colors'])
        # obj_embeds is (batch_size, num_objects, hidden_size)
        # 文本编码
        txt_embeds = self.txt_encoder(
            batch['txt_ids'], batch['txt_masks'],
        ).last_hidden_state
        # txt_embeds is (batch_size, max_txt_len, hidden_size)
        if self.config.txt_encoder.freeze:
            txt_embeds = txt_embeds.detach()

        # 多模态编码
        out_embeds = self.mm_encoder(
            txt_embeds, batch['txt_masks'],
            obj_embeds, batch['obj_locs'], batch['obj_masks'],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # obj_embeds: (batch_size, num_objects, hidden_size)，all_cross_attns: List of (batch_size, num_heads, num_objects, max_txt_len)，all_self_attns: List of (batch_size, num_heads, num_objects, num_objects)，all_hidden_states: List of (batch_size, num_objects, hidden_size)
        # 计算目标3D的对数几率
        og3d_logits = self.og3d_head(out_embeds['obj_embeds']).squeeze(2)  # 输出(batch_size, num_objects*p)
        og3d_logits.masked_fill_(batch['obj_masks'].logical_not(), -float('inf'))  # 将掩码为False的位置设置为负无穷，以便在计算损失时忽略这些位置
        result = {
            'og3d_logits': og3d_logits,
        }
        if output_attentions:
            result['all_cross_attns'] = out_embeds['all_cross_attns']
            result['all_self_attns'] = out_embeds['all_self_attns']
        if output_hidden_states:
            result['all_hidden_states'] = out_embeds['all_hidden_states']

        # 如果配置中指定了损失函数的权重，则计算相应的损失
        if self.config.losses.obj3d_clf > 0:
            result['obj3d_clf_logits'] = self.obj3d_clf_head(
                out_embeds['obj_embeds'])  # 输出(batch_size, num_objects, num_obj_classes)
        if self.config.losses.obj3d_reg > 0:
            result['obj3d_loc_preds'] = self.obj3d_reg_head(out_embeds['obj_embeds']) # 输出(batch_size, num_objects, 3)
        if self.config.losses.obj3d_clf_pre > 0:
            result['obj3d_clf_pre_logits'] = self.obj3d_clf_pre_head(obj_embeds) # 输出(batch_size, num_objects, num_obj_classes)
        if self.config.losses.txt_clf > 0:
            result['txt_clf_logits'] = self.txt_clf_head(txt_embeds[:, 0])  # 取第一个标记（通常是 [CLS] 标记）用于聚合整个序列的信息

        if compute_loss:
            losses = self.compute_loss(result, batch)
            return result, losses

        return result

    # 计算损失函数
    def compute_loss(self, result, batch):
        losses = {}
        total_loss = 0

        # 计算目标3D的分类损失
        og3d_loss = F.cross_entropy(result['og3d_logits'], batch['tgt_obj_idxs'])
        losses['og3d'] = og3d_loss
        total_loss += og3d_loss

        # 如果配置中指定了损失函数的权重，则计算相应的损失
        if self.config.losses.obj3d_clf > 0:
            obj3d_clf_loss = F.cross_entropy(
                result['obj3d_clf_logits'].permute(0, 2, 1),  # 调整张量的维度顺序，以便符合损失函数 F.cross_entropy 的输入要求
                batch['obj_classes'], reduction='none'
            )
            obj3d_clf_loss = (obj3d_clf_loss * batch['obj_masks']).sum() / batch['obj_masks'].sum() #将计算得到的损失与对象掩码相乘，以忽略那些被掩码掉的对象
            losses['obj3d_clf'] = obj3d_clf_loss * self.config.losses.obj3d_clf
            total_loss += losses['obj3d_clf']

        if self.config.losses.obj3d_clf_pre > 0:
            obj3d_clf_pre_loss = F.cross_entropy(
                result['obj3d_clf_pre_logits'].permute(0, 2, 1),
                batch['obj_classes'], reduction='none'
            )
            obj3d_clf_pre_loss = (obj3d_clf_pre_loss * batch['obj_masks']).sum() / batch['obj_masks'].sum()
            losses['obj3d_clf_pre'] = obj3d_clf_pre_loss * self.config.losses.obj3d_clf_pre
            total_loss += losses['obj3d_clf_pre']

        if self.config.losses.obj3d_reg > 0:
            obj3d_reg_loss = F.mse_loss(
                result['obj3d_loc_preds'], batch['obj_locs'][:, :, :3], reduction='none'
            )
            obj3d_reg_loss = (obj3d_reg_loss * batch['obj_masks'].unsqueeze(2)).sum() / batch['obj_masks'].sum() #unsqueeze(2)用于增加一个维度以匹配obj3d_reg_loss
            losses['obj3d_reg'] = obj3d_reg_loss * self.config.losses.obj3d_reg
            total_loss += losses['obj3d_reg']

        if self.config.losses.txt_clf > 0:
            txt_clf_loss = F.cross_entropy(
                result['txt_clf_logits'], batch['tgt_obj_classes'], reduction='mean'
            )
            losses['txt_clf'] = txt_clf_loss * self.config.losses.txt_clf
            total_loss += losses['txt_clf']

        losses['total'] = total_loss
        return losses
