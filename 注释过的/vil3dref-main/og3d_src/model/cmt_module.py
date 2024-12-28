'''
定义了一个基于Transformer架构的模型，用于处理序列数据，特别是在处理空间关系方面进行了扩展。
模型包括一个变换器解码层（TransformerDecoderLayer），一个多头部注意力机制的扩展（MultiHeadAttentionSpatial），
以及一个完整的模型（CMT），用于结合文本和空间信息进行预测。
'''
import copy
import numpy as np
from typing import Optional
import time

import einops

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # 激活函数
        self.activation = _get_activation_fn(activation)

    def forward(
            self, tgt, memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ):
        # 自注意力
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                                                  key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        # 编码器-解码器注意力
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(query=tgt2, key=memory, value=memory, attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        # 前馈网络
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices


# 定义一个多头注意力机制的扩展，用于处理空间信息
class MultiHeadAttentionSpatial(nn.Module):
    """
    多头注意力机制与空间信息融合的模块。

    该模块在传统的多头注意力机制基础上，增加了对空间信息的处理，通过不同的策略将空间信息融入到注意力机制中，以提升模型对空间位置的敏感度。

    参数:
    - d_model: 模型的维度。
    - n_head: 多头注意力的头数。
    - dropout: Dropout的概率，默认为0.1。
    - spatial_multihead: 是否为每个注意力头分配独立的空间信息，默认为True。
    - spatial_dim: 空间信息的维度，默认为5。
    - spatial_attn_fusion: 空间信息与注意力机制的融合方式，默认为'mul'，支持['mul', 'bias', 'add', 'ctx', 'cond']。
    """

    def __init__(
            self, d_model, n_head, dropout=0.1, spatial_multihead=True, spatial_dim=5,
            spatial_attn_fusion='mul',
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' % (d_model, n_head)  # 确保模型维度可以被头数整除

        # 初始化模型参数
        self.n_head = n_head  # 注意力头数
        self.d_model = d_model  # 模型维度
        self.d_per_head = d_model // n_head  # 每个头的维度
        self.spatial_multihead = spatial_multihead  # 是否为每个注意力头分配独立的空间信息
        self.spatial_dim = spatial_dim  # 空间信息的维度
        self.spatial_attn_fusion = spatial_attn_fusion  # 空间信息与注意力机制的融合方式

        # 查询、键、值的线性变换
        self.w_qs = nn.Linear(d_model, d_model)  # 查询的线性变换
        self.w_ks = nn.Linear(d_model, d_model)  # 键的线性变换
        self.w_vs = nn.Linear(d_model, d_model)  # 值的线性变换

        # 输出的线性变换、dropout及层归一化
        self.fc = nn.Linear(d_model, d_model)  # 输出的线性变换
        self.dropout = nn.Dropout(p=dropout)  # Dropout层
        self.layer_norm = nn.LayerNorm(d_model)  # 层归一化

        # 空间注意力融合
        self.spatial_n_head = n_head if spatial_multihead else 1  # 空间注意力头数
        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)  # 位置信息的线性变换
        elif self.spatial_attn_fusion == 'ctx':
            self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)  # 位置信息的线性变换
        elif self.spatial_attn_fusion == 'cond':
            self.lang_cond_fc = nn.Linear(d_model, self.spatial_n_head * (spatial_dim + 1))  # 条件空间注意力的线性变换
        else:
            raise NotImplementedError('unsupported spatial_attn_fusion %s' % (self.spatial_attn_fusion))  # 抛出不支持的融合方式异常

    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
        """
        前向传播函数。

        参数:
        - q: 查询矩阵。
        - k: 键矩阵。
        - v: 值矩阵。
        - pairwise_locs: 成对位置信息。
        - key_padding_mask: 键的padding掩码，可选。
        - txt_embeds: 文本嵌入，用于条件空间注意力融合，可选。

        返回:
        - output: 注意力机制的输出。
        - fused_attn: 融合后的注意力权重。
        """
        residual = q  # 保存输入的查询矩阵，用于残差连接

        # 对查询、键、值进行线性变换，并重塑为多头形式
        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)  # 查询的线性变换并重塑
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)  # 键的线性变换并重塑
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)  # 值的线性变换并重塑

        # 计算注意力得分
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])  # 计算注意力得分并缩放

        # 根据不同的空间注意力融合策略，计算空间注意力
        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            loc_attn = self.pairwise_loc_fc(pairwise_locs)  # 通过全连接层计算位置注意力
            loc_attn = einops.rearrange(loc_attn, 'b l t h -> h b l t')  # 重塑位置注意力张量
            if self.spatial_attn_fusion == 'mul':
                loc_attn = F.relu(loc_attn)  # 应用ReLU激活函数
            if not self.spatial_multihead:
                loc_attn = einops.repeat(loc_attn, 'h b l t -> (h nh) b l t', nh=self.n_head)  # 重复位置注意力张量
        elif self.spatial_attn_fusion == 'ctx':
            loc_attn = self.pairwise_loc_fc(pairwise_locs)  # 通过全连接层计算位置注意力
            loc_attn = einops.rearrange(loc_attn, 'b l t (h k) -> h b l t k', h=self.n_head)  # 重塑位置注意力张量
            loc_attn = torch.einsum('hblk,hbltk->hblt', q, loc_attn) / np.sqrt(q.shape[-1])  # 计算注意力得分并缩放
        elif self.spatial_attn_fusion == 'cond':
            spatial_weights = self.lang_cond_fc(residual + txt_embeds.unsqueeze(1))  # 计算条件空间权重
            spatial_weights = einops.rearrange(spatial_weights, 'b l (h d) -> h b l d', h=self.spatial_n_head,
                                               d=self.spatial_dim + 1)  # 重塑条件空间权重
            if self.spatial_n_head == 1:
                spatial_weights = einops.repeat(spatial_weights, '1 b l d -> h b l d', h=self.n_head)  # 重复条件空间权重
            spatial_bias = spatial_weights[..., :1]  # 提取偏置项
            spatial_weights = spatial_weights[..., 1:]  # 提取权重项
            loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights, pairwise_locs) + spatial_bias  # 计算位置注意力
            loc_attn = torch.sigmoid(loc_attn)  # 应用Sigmoid激活函数

        # 如果提供了键的padding掩码，则应用掩码
        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))  # 生成掩码
            attn = attn.masked_fill(mask, -np.inf)  # 应用掩码到注意力得分
            if self.spatial_attn_fusion in ['mul', 'cond']:
                loc_attn = loc_attn.masked_fill(mask, 0)  # 应用掩码到位置注意力
            else:
                loc_attn = loc_attn.masked_fill(mask, -np.inf)  # 应用掩码到位置注意力

        # 根据不同的空间注意力融合策略，融合注意力得分与空间注意力
        if self.spatial_attn_fusion == 'add':
            fused_attn = (torch.softmax(attn, 3) + torch.softmax(loc_attn, 3)) / 2  # 融合注意力得分与位置注意力
        else:
            if self.spatial_attn_fusion in ['mul', 'cond']:
                fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn  # 融合注意力得分与位置注意力
            else:
                fused_attn = loc_attn + attn  # 融合注意力得分与位置注意力
            fused_attn = torch.softmax(fused_attn, 3)  # 应用Softmax激活函数

        # 确保注意力权重中没有NaN值
        assert torch.sum(torch.isnan(fused_attn) == 0), print(fused_attn)  # 检查注意力权重

        # 计算注意力输出
        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)  # 计算注意力输出
        output = einops.rearrange(output, 'head b l v -> b l (head v)')  # 重塑输出张量
        output = self.dropout(self.fc(output))  # 应用线性变换和Dropout
        output = self.layer_norm(output + residual)  # 应用残差连接和层归一化

        return output, fused_attn  # 返回注意力机制的输出和融合后的注意力权重


class TransformerSpatialDecoderLayer(TransformerDecoderLayer):
    def __init__(
            self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
            spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        super().__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        # 移除原来的自注意力机制
        del self.self_attn
        # 使用空间多头注意力机制替代原来的自注意力机制
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout,
            spatial_multihead=spatial_multihead,
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

    def forward(
            self, tgt, memory, tgt_pairwise_locs,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ):
        # 正常的前向传播过程，但是使用了空间多头注意力机制
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, tgt2, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask,
            txt_embeds=memory[:, 0], # 将文本嵌入传递给空间注意力机制
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices


class CMT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 根据配置决定使用哪种解码层
        if self.config.spatial_dec:
            decoder_class = TransformerSpatialDecoderLayer
            kwargs = {
                'spatial_dim': config.spatial_dim,
                'spatial_multihead': config.spatial_multihead,
                'spatial_attn_fusion': config.spatial_attn_fusion,
            }
        else:
            decoder_class = TransformerDecoderLayer
            kwargs = {}

        # 创建解码层实例并复制多层
        decoder_layer = decoder_class(
            config.hidden_size, config.num_attention_heads,
            dim_feedforward=2048, dropout=0.1, activation='gelu', **kwargs
        )
        self.layers = _get_clones(decoder_layer, config.num_layers)

        # 创建位置编码层
        loc_layer = nn.Sequential(
            nn.Linear(config.dim_loc, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )
        # 根据配置决定创建多少层位置编码层
        if self.config.obj_loc_encoding in ['same_0', 'same_all']:
            num_loc_layers = 1
        elif self.config.obj_loc_encoding == 'diff_all':
            num_loc_layers = config.num_layers
        self.loc_layers = _get_clones(loc_layer, num_loc_layers)

        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 初始化线性层的权重，使用均值为0，标准差为0.02的正态分布
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 如果线性层有偏置项，则将偏置项初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层的权重，使用均值为0，标准差为0.02的正态分布
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 如果嵌入层有填充索引，则将填充索引对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化层归一化的偏置项为0
            module.bias.data.zero_()
            # 初始化层归一化的权重为1
            module.weight.data.fill_(1.0)

    def calc_pairwise_locs(self, obj_centers, obj_whls, eps=1e-10, pairwise_rel_type='center'):
        """
        计算物体之间的相对位置。

        参数:
            obj_centers (torch.Tensor): 物体的中心点坐标，形状为 [batch_size, num_objs, dim]。
            obj_whls (torch.Tensor): 物体的宽高深，形状为 [batch_size, num_objs, 3]。
            eps (float): 用于防止除零错误的小值。
            pairwise_rel_type (str): 计算相对位置的类型，可以是 'center' 或 'vertical_bottom'。

        返回:
            torch.Tensor: 物体之间的相对位置信息。
        """

        # 计算物体之间的相对位置
        if pairwise_rel_type == 'mlp':
            # 使用 MLP 来计算相对位置
            obj_locs = torch.cat([obj_centers, obj_whls], 2) #拼接对象中心点和宽高信息：将对象的中心点坐标和宽高信息拼接在一起，形成 obj_locs
            pairwise_locs = torch.cat(
                [
                    einops.repeat(obj_locs, 'b l d -> b l x d', x=obj_locs.size(1)),
                    einops.repeat(obj_locs, 'b l d -> b x l d', x=obj_locs.size(1))
                ],
                dim=3 #通过 einops.repeat 方法，将 obj_locs 在两个不同的维度上重复，然后拼接起来，形成 pairwise_locs。
            )
            return pairwise_locs

        # 计算物体中心之间的相对位置
        pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l1 d') \
                        - einops.repeat(obj_centers, 'b l d -> b 1 l d')
        pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 3) + eps)  # 计算欧氏距离

        # 归一化距离，使数值更稳定
        if self.config.spatial_dist_norm:
            max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
            norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, 'b -> b 1 1')
        else:
            norm_pairwise_dists = pairwise_dists

        # 如果只需要一维距离信息
        if self.config.spatial_dim == 1:
            return norm_pairwise_dists.unsqueeze(3)

        # 计算2D距离
        pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)

        # 根据相对位置类型选择计算方式
        if pairwise_rel_type == 'center':
            pairwise_locs = torch.stack(
                [
                    norm_pairwise_dists, pairwise_locs[..., 2] / pairwise_dists,
                                         pairwise_dists_2d / pairwise_dists, pairwise_locs[..., 1] / pairwise_dists_2d,
                                         pairwise_locs[..., 0] / pairwise_dists_2d
                ],
                dim=3
            )
        elif pairwise_rel_type == 'vertical_bottom':
            bottom_centers = torch.clone(obj_centers)
            bottom_centers[:, :, 2] -= obj_whls[:, :, 2]  # 计算底部中心
            bottom_pairwise_locs = einops.repeat(bottom_centers, 'b l d -> b l 1 d') \
                                   - einops.repeat(bottom_centers, 'b l d -> b 1 l d')
            bottom_pairwise_dists = torch.sqrt(torch.sum(bottom_pairwise_locs ** 2, 3) + eps)
            bottom_pairwise_dists_2d = torch.sqrt(torch.sum(bottom_pairwise_locs[..., :2] ** 2, 3) + eps)
            pairwise_locs = torch.stack(
                [
                    norm_pairwise_dists,
                    bottom_pairwise_locs[..., 2] / bottom_pairwise_dists,
                    bottom_pairwise_dists_2d / bottom_pairwise_dists,
                    pairwise_locs[..., 1] / pairwise_dists_2d,
                    pairwise_locs[..., 0] / pairwise_dists_2d
                ],
                dim=3
            )

        # 根据空间维度配置，调整输出维度
        if self.config.spatial_dim == 4:
            pairwise_locs = pairwise_locs[..., 1:]
        return pairwise_locs

    def forward(
            self, txt_embeds, txt_masks, obj_embeds, obj_locs, obj_masks,
            output_attentions=False, output_hidden_states=False,
    ):
        """
        执行模型的前向传播。

        参数:
            txt_embeds (torch.Tensor): 文本嵌入。
            txt_masks (torch.Tensor): 文本掩码，用于指示哪些位置是有效的。
            obj_embeds (torch.Tensor): 对象嵌入。
            obj_locs (torch.Tensor): 对象的位置信息。
            obj_masks (torch.Tensor): 对象掩码，用于指示哪些对象是有效的。
            output_attentions (bool): 是否输出注意力矩阵。
            output_hidden_states (bool): 是否输出隐藏状态。

        返回:
            dict: 包含对象嵌入和其他可能的输出。
        """

        # 如果配置中启用了空间解码，计算对象之间的相对位置
        if self.config.spatial_dec:
            pairwise_locs = self.calc_pairwise_locs(
                obj_locs[:, :, :3], obj_locs[:, :, 3:],
                pairwise_rel_type=self.config.pairwise_rel_type
            )

        # 初始化输出嵌入为对象嵌入
        out_embeds = obj_embeds
        # 存储每层的输出嵌入
        all_hidden_states = [out_embeds]
        # 存储每层的自注意力矩阵和交叉注意力矩阵
        all_self_attn_matrices, all_cross_attn_matrices = [], []

        # 遍历模型的每一层
        for i, layer in enumerate(self.layers):
            # 如果配置中指定每个对象位置编码不同，则对每层应用位置编码
            if self.config.obj_loc_encoding == 'diff_all':
                query_pos = self.loc_layers[i](obj_locs)
                out_embeds = out_embeds + query_pos
            else:
                # 如果配置中指定所有对象位置编码相同，则只在第一层应用位置编码
                query_pos = self.loc_layers[0](obj_locs)
                if self.config.obj_loc_encoding == 'same_all':
                    out_embeds = out_embeds + query_pos
                else:
                    # 如果只在第一层应用位置编码
                    if i == 0:
                        out_embeds = out_embeds + query_pos

            # 如果启用了空间解码，将相对位置信息传递给层
            if self.config.spatial_dec:
                out_embeds, self_attn_matrices, cross_attn_matrices = layer(
                    out_embeds, txt_embeds, pairwise_locs,
                    tgt_key_padding_mask=obj_masks.logical_not(),
                    memory_key_padding_mask=txt_masks.logical_not(),
                )
            else:
                # 否则，只传递文本嵌入
                out_embeds, self_attn_matrices, cross_attn_matrices = layer(
                    out_embeds, txt_embeds,
                    tgt_key_padding_mask=obj_masks.logical_not(),
                    memory_key_padding_mask=txt_masks.logical_not(),
                )

            # 存储每层的输出
            all_hidden_states.append(out_embeds)
            all_self_attn_matrices.append(self_attn_matrices)
            all_cross_attn_matrices.append(cross_attn_matrices)

        # 构建输出字典
        outs = {
            'obj_embeds': out_embeds,
        }
        # 如果需要，添加隐藏状态和注意力矩阵到输出
        if output_hidden_states:
            outs['all_hidden_states'] = all_hidden_states
        if output_attentions:
            outs['all_self_attns'] = all_self_attn_matrices
            outs['all_cross_attns'] = all_cross_attn_matrices
        return outs
