import os
import jsonlines
import json
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from .common import pad_tensors, gen_seq_masks
    from .gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES
except:
    from common import pad_tensors, gen_seq_masks
    from gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES

class GTLabelPcdDataset(GTLabelDataset):
    '''用于处理与点云数据（PCD）相关的数据，这些数据通常用于三维场景理解任务'''
    def __init__(
        self, scan_id_file, anno_file, scan_dir, category_file,
        cat2vec_file=None, keep_background=False, random_rotate=False,
        num_points=1024, max_txt_len=50, max_obj_len=80,
        in_memory=False, gt_scan_dir=None, iou_replace_gt=0
    ):
        '''
        初始化 GTLabelPcdDataset 类的实例。

        参数:
        - scan_id_file: 扫描ID文件的路径。
        - anno_file: 注释文件的路径。
        - scan_dir: 扫描目录的路径。
        - category_file: 类别文件的路径。
        - cat2vec_file: 类别向量文件的路径（可选）。
        - keep_background: 是否保留背景（可选）。
        - random_rotate: 是否随机旋转点云（可选）。
        - num_points: 每个对象的点数（可选）。
        - max_txt_len: 文本的最大长度（可选）。
        - max_obj_len: 对象的最大长度（可选）。
        - in_memory: 是否将点云数据加载到内存中（可选）。
        - gt_scan_dir: 地面真实扫描目录的路径（可选）。
        - iou_replace_gt: IOU阈值，用于决定是否用预测的边界框替换地面真实的边界框（可选）。
        '''
        # 调用父类的初始化方法
        super().__init__(
            scan_id_file, anno_file, scan_dir, category_file,
            cat2vec_file=cat2vec_file, keep_background=keep_background,
            random_rotate=random_rotate, 
            max_txt_len=max_txt_len, max_obj_len=max_obj_len,
            gt_scan_dir=gt_scan_dir, iou_replace_gt=iou_replace_gt
        )
        # 初始化点云相关的成员变量
        self.num_points = num_points
        self.in_memory = in_memory
        # 如果设置为将点云数据加载到内存中，则预先加载点云数据
        if self.in_memory:
            for scan_id in self.scan_ids:
                self.get_scan_pcd_data(scan_id)

    def get_scan_pcd_data(self, scan_id):
        """
                加载扫描的点云数据。

                参数:
                - scan_id: 扫描的ID。

                返回:
                - obj_pcds: 对象的点云数据列表。
                """
        # 如果点云数据已经在内存中，则直接返回
        if self.in_memory and 'pcds' in self.scans[scan_id]:
            return self.scans[scan_id]['pcds']
        # 加载点云数据
        pcd_data = torch.load(
            os.path.join(self.scan_dir, 'pcd_with_global_alignment', '%s.pth'%scan_id)
        )
        points, colors = pcd_data[0], pcd_data[1]
        colors = colors / 127.5 - 1 # 归一化颜色值
        pcds = np.concatenate([points, colors], 1) # 合并点和颜色
        instance_labels = pcd_data[-1] # 实例标签
        obj_pcds = []
        for i in range(instance_labels.max() + 1):  # 对每个实例
            mask = instance_labels == i  # 选择当前实例的点
            obj_pcds.append(pcds[mask])  # 添加到对象点云列表
        # 如果设置为将点云数据加载到内存中，则存储点云数据
        if self.in_memory:
            self.scans[scan_id]['pcds'] = obj_pcds
        return obj_pcds

    def get_scan_gt_pcd_data(self, scan_id):
        """
        加载扫描的地面真实点云数据。

        参数:
        - scan_id: 扫描的ID。

        返回:
        - obj_pcds: 对象的真实点云数据列表。
        """

        # 如果真实点云数据已经在内存中，则直接返回

        if self.in_memory and 'gt_pcds' in self.scans[scan_id]:
            return self.scans[scan_id]['gt_pcds']
        # 加载真实点云数据
        pcd_data = torch.load(
            os.path.join(self.gt_scan_dir, 'pcd_with_global_alignment', '%s.pth'%scan_id)
        )
        points, colors = pcd_data[0], pcd_data[1]
        colors = colors / 127.5 - 1  # 归一化颜色值
        pcds = np.concatenate([points, colors], 1)  # 合并点和颜色 包含点云和颜色的标签
        instance_labels = pcd_data[-1]  # 实例标签
        obj_pcds = []
        for i in range(instance_labels.max() + 1):  # 对每个实例
            mask = instance_labels == i  # 选择当前实例的点
            obj_pcds.append(pcds[mask])  # 添加到对象点云列表
        # 如果设置为将点云数据加载到内存中，则存储点云数据
        if self.in_memory:
            self.scans[scan_id]['gt_pcds'] = obj_pcds
        return obj_pcds

    def _get_obj_inputs(self, obj_pcds, obj_colors, obj_labels, obj_ids, tgt_obj_idx, theta=None):
        """
                处理对象输入，包括旋转和选择对象。

                参数:
                - obj_pcds: 对象的点云数据列表。
                - obj_colors: 对象的颜色列表。
                - obj_labels: 对象的标签列表。
                - obj_ids: 对象的ID列表。
                - tgt_obj_idx: 目标对象的索引。
                - theta: 旋转角度（可选）。

                返回:
                - obj_fts: 处理后的对象特征。
                - obj_locs: 处理后的对象位置。
                - obj_colors: 处理后的对象颜色。
                - obj_labels: 处理后的对象标签。
                - obj_ids: 处理后的对象ID。
                - tgt_obj_idx: 处理后的目标对象索引。
                """
        # 如果对象数量超过最大长度，则选择与目标对象相同类型的其他对象
        tgt_obj_type = obj_labels[tgt_obj_idx]
        if (self.max_obj_len is not None) and (len(obj_labels) > self.max_obj_len):
            selected_obj_idxs = [tgt_obj_idx]
            remained_obj_idxs = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj != tgt_obj_idx:
                    if klabel == tgt_obj_type:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idxs.append(kobj)
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idxs)
                selected_obj_idxs += remained_obj_idxs[:self.max_obj_len - len(selected_obj_idxs)]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]
            tgt_obj_idx = 0

            # 如果需要旋转，则创建旋转矩阵
            if (theta is not None) and (theta != 0):
                rot_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                rot_matrix = None

            obj_fts, obj_locs = [], []
            for obj_pcd in obj_pcds:
                # 如果需要旋转，则旋转点云数据
                if rot_matrix is not None:
                    obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
                obj_center = obj_pcd[:, :3].mean(0)
                obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                obj_locs.append(np.concatenate([obj_center, obj_size], 0))
                # 采样点云数据
                pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points,
                                            replace=(len(obj_pcd) < self.num_points))
                obj_pcd = obj_pcd[pcd_idxs]
                # 归一化点云数据
                obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
                max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3] ** 2, 1)))
                if max_dist < 1e-6:  # 处理微小点云数据
                    max_dist = 1
                obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
                obj_fts.append(obj_pcd)

        obj_fts = np.stack(obj_fts, 0) #按行堆叠
        obj_locs = np.array(obj_locs)
        obj_colors = np.array(obj_colors)
            
        return obj_fts, obj_locs, obj_colors, obj_labels, obj_ids, tgt_obj_idx #obj_fts：(num_objects, num_points, feature_dim)，obj_locs：(num_objects, 6)，obj_colors：(num_objects, color_dim)，obj_labels：(num_objects,)，obj_ids：(num_objects,)，tgt_obj_idx：标量

    def __getitem__(self, idx):
        """
                获取单个数据点。

                参数:
                - idx: 数据点的索引。

                返回:
                - 一个包含处理后的数据点的字典。
                """
        item = self.data[idx]
        scan_id = item['scan_id']
        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        tgt_obj_idx = item['target_id']
        tgt_obj_type = item['instance_type']

        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        txt_lens = len(txt_tokens)

        # 加载点云数据
        if self.gt_scan_dir is None or item['max_iou'] > self.iou_replace_gt:
            obj_pcds = self.get_scan_pcd_data(scan_id)
            obj_labels = self.scans[scan_id]['inst_labels']
            obj_gmm_colors = self.scans[scan_id]['inst_colors']
        else:
            tgt_obj_idx = item['gt_target_id']
            obj_pcds = self.get_scan_gt_pcd_data(scan_id)
            obj_labels = self.scans[scan_id]['gt_inst_labels']
            obj_gmm_colors = self.scans[scan_id]['gt_inst_colors']
        obj_ids = [str(x) for x in range(len(obj_labels))]

        # 如果不保留背景，则过滤背景对象
        if not self.keep_background:
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if obj_label not in ['wall', 'floor', 'ceiling']]
            tgt_obj_idx = selected_obj_idxs.index(tgt_obj_idx)
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_gmm_colors = [obj_gmm_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]

        # 如果需要随机旋转，则随机选择一个旋转角度
        if self.random_rotate:
            theta_idx = np.random.randint(len(ROTATE_ANGLES))
            theta = ROTATE_ANGLES[theta_idx]
        else:
            theta = 0

        # 处理对象输入
        aug_obj_fts, aug_obj_locs, aug_obj_gmm_colors, aug_obj_labels, \
            aug_obj_ids, aug_tgt_obj_idx = self._get_obj_inputs(
                obj_pcds, obj_gmm_colors, obj_labels, obj_ids, tgt_obj_idx,
                theta=theta
            )

        # 将 NumPy 数据转换为 PyTorch 张量
        aug_obj_fts = torch.from_numpy(aug_obj_fts)
        aug_obj_locs = torch.from_numpy(aug_obj_locs)
        aug_obj_gmm_colors = torch.from_numpy(aug_obj_gmm_colors)
        aug_obj_classes = torch.LongTensor([self.cat2int[x] for x in aug_obj_labels])

        # 如果有类别向量，则使用类别向量
        if self.cat2vec is None:
            aug_obj_gt_fts = aug_obj_classes
        else:
            aug_obj_gt_fts = torch.FloatTensor([self.cat2vec[x] for x in aug_obj_labels])

        outs = {
            'item_ids': item['item_id'],
            'scan_ids': scan_id,
            'txt_ids': txt_tokens,
            'txt_lens': txt_lens,
            'obj_gt_fts': aug_obj_gt_fts,
            'obj_fts': aug_obj_fts,
            'obj_locs': aug_obj_locs,
            'obj_colors': aug_obj_gmm_colors,
            'obj_lens': len(aug_obj_fts),
            'obj_classes': aug_obj_classes, 
            'tgt_obj_idxs': aug_tgt_obj_idx,
            'tgt_obj_classes': self.cat2int[tgt_obj_type],
            'obj_ids': aug_obj_ids,
        }
        return outs


def gtlabelpcd_collate_fn(data):
    """
    自定义collate_fn函数，用于处理数据集中的数据项并返回一个批次的数据。

    此函数的主要目的是对输入的data进行处理，包括：
    1. 根据数据项的keys组织输出数据的字典结构。
    2. 对文本和对象相关的数据进行填充和张量化处理，以适应模型的输入要求。
    3. 生成必要的掩码张量，以处理变长序列数据。

    参数:
    - data: 一个列表，包含一个批次的数据项。每个数据项通常是一个由多个特征组成的字典。

    返回:
    - outs: 一个字典，包含处理后的批次数据。数据类型包括列表、张量等。
    """
    # 初始化输出字典，以适应数据项的结构
    outs = {}
    # 遍历数据项中的所有keys，组织输出数据的结构
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]

    # 对文本ID进行填充，确保序列长度一致
    outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
    # 将文本长度转换为LongTensor类型
    outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
    # 根据文本长度生成掩码张量
    outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])

    # 对对象的特征张量进行填充，处理变长序列数据
    outs['obj_gt_fts'] = pad_tensors(outs['obj_gt_fts'], lens=outs['obj_lens'])
    outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'], pad_ori_data=True)
    outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0)
    outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0)
    # 将对象长度转换为LongTensor类型
    outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
    # 根据对象长度生成掩码张量
    outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])

    # 对对象类别进行填充，使用-100作为padding值，以适应损失计算的需要
    outs['obj_classes'] = pad_sequence(
        outs['obj_classes'], batch_first=True, padding_value=-100
    )
    # 将目标对象的索引和类别转换为LongTensor类型
    outs['tgt_obj_idxs'] = torch.LongTensor(outs['tgt_obj_idxs'])
    outs['tgt_obj_classes'] = torch.LongTensor(outs['tgt_obj_classes'])

    # 返回处理后的批次数据
    return outs


