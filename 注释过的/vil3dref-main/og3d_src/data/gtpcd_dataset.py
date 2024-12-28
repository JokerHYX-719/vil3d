import os
import jsonlines
import json
import numpy as np
import random
import time

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from .common import pad_tensors, gen_seq_masks
    from .gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES
except:
    from common import pad_tensors, gen_seq_masks
    from gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES

class GTPcdDataset(GTLabelDataset):
    '''数据集类 GTPcdDataset 是基于 GTLabelDataset 的一个子类，专门用于处理点云数据（Point Cloud Data，简称PCD）'''
    def __init__(
        self, scan_id_file, anno_file, scan_dir, category_file,
        cat2vec_file=None, keep_background=False, 
        num_points=1024, max_txt_len=50, max_obj_len=80,
        random_rotate=False, in_memory=False
    ):
        '''
        scan_id_file：包含扫描ID的文件路径，标识了数据集中包含哪些扫描。
        anno_file：注释文件的路径，包含了文本描述和其他元数据。
        scan_dir：扫描数据的目录路径。
        category_file：类别文件的路径，包含了类别到索引的映射。
        cat2vec_file：可选的，类别到向量的映射文件路径。
        keep_background：是否保留背景对象。
        num_points：每个点云样本的点数。
        max_txt_len 和 max_obj_len：文本和对象标注的最大长度。
        random_rotate：是否对数据进行随机旋转增强。
        in_memory：是否将数据加载到内存中，以加快数据加载速度
        '''
        super().__init__(
            scan_id_file, anno_file, scan_dir, category_file,
            cat2vec_file=cat2vec_file, keep_background=keep_background,
            random_rotate=random_rotate, 
            max_txt_len=max_txt_len, max_obj_len=max_obj_len,
        )
        self.num_points = num_points
        self.in_memory = in_memory

        if self.in_memory:
            for scan_id in self.scan_ids:
                self.get_scan_pcd_data(scan_id)

    def get_scan_pcd_data(self, scan_id):
        '''
        加载和处理单个扫描的点云数据。它从文件中加载点云数据，然后根据实例标签将点云分割成单独的对象点云。

        参数:
        scan_id: 扫描的唯一标识符，用于识别特定的扫描数据。

        返回:
        obj_pcds: 一个列表，包含根据实例标签分割后的对象点云数据。
        '''
        # 如果数据已经在内存中，则直接返回，避免重复加载
        if self.in_memory and 'pcds' in self.scans[scan_id]:
            return self.scans[scan_id]['pcds']

        # 从文件系统中加载点云数据
        pcd_data = torch.load(
            os.path.join(self.scan_dir, 'pcd_with_global_alignment', '%s.pth' % scan_id)
        )
        # 分离点云数据中的点坐标和颜色信息
        points, colors = pcd_data[0], pcd_data[1]
        # 对颜色信息进行归一化处理
        colors = colors / 127.5 - 1
        # 将点坐标和颜色信息合并，以便后续处理
        pcds = np.concatenate([points, colors], 1)

        # 获取实例标签，用于分割点云
        instance_labels = pcd_data[-1]
        obj_pcds = []
        # 根据实例标签，将点云分割成单独的对象点云
        for i in range(instance_labels.max() + 1):
            # 创建一个掩码，用于提取当前实例的所有点
            mask = instance_labels == i  # time consuming
            # 使用掩码提取当前实例的点云数据，并添加到列表中
            obj_pcds.append(pcds[mask])

        # 如果设置为在内存中加载数据，则将处理后的数据保存在内存中，以备后续快速访问
        if self.in_memory:
            self.scans[scan_id]['pcds'] = obj_pcds

        # 返回处理后的对象点云数据
        return obj_pcds

    def _get_obj_inputs(self, obj_pcds, obj_colors, obj_labels, obj_ids, tgt_obj_idx, theta=None):
        '''
        处理对象的输入数据，包括选择对象、应用随机旋转、采样点云、归一化点云位置等。它还计算对象的中心位置和大小，并将这些信息作为对象的特征。

        参数:
        - obj_pcds: 对象的点云数据列表。
        - obj_colors: 对象的颜色数据列表。
        - obj_labels: 对象的标签列表。
        - obj_ids: 对象的ID列表。
        - tgt_obj_idx: 目标对象的索引。
        - theta: 旋转角度，如果提供，则对对象应用此旋转。

        返回:
        - obj_fts: 处理后的对象特征数组。
        - obj_locs: 对象的位置和大小信息数组。
        - obj_colors: 处理后的对象颜色数组。
        - obj_labels: 处理后的对象标签列表。
        - obj_ids: 处理后的对象ID列表。
        - tgt_obj_idx: 处理后的目标对象索引。
        '''
        # 获取目标对象的类型
        tgt_obj_type = obj_labels[tgt_obj_idx]

        # 如果设置了最大对象数量且当前对象数量超过这个值，则进行对象选择
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

        # 根据提供的旋转角度生成旋转矩阵
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
            # 应用旋转矩阵到点云数据
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            # 计算对象的中心位置和大小
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # 从点云中随机选择固定数量的点
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
            obj_pcd = obj_pcd[pcd_idxs]

            # 对点云位置进行归一化（这个是label 没有的）
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3] ** 2, 1)))
            if max_dist < 1e-6:  # 处理极小的点云，即填充情况
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist

            obj_fts.append(obj_pcd)

        # 将处理后的对象特征和位置信息转换为数组形式
        obj_fts = np.stack(obj_fts, 0)
        obj_locs = np.array(obj_locs)
        obj_colors = np.array(obj_colors)

        # 返回处理后的数据
        return obj_fts, obj_locs, obj_colors, obj_labels, obj_ids, tgt_obj_idx

    def __getitem__(self, idx):
        """
        根据索引获取数据项。

        参数:
        - idx: 索引值，用于获取数据项。

        返回:
        - outs: 包含各项数据的字典，如文本ID、扫描ID、对象特征等。
        """
        # 从数据集中获取指定索引的项
        item = self.data[idx]
        # 提取扫描ID、目标对象索引和目标对象类型
        scan_id = item['scan_id']
        tgt_obj_idx = item['target_id']
        tgt_obj_type = item['instance_type']

        # 将文本标记转换为张量，并限制其长度
        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        txt_lens = len(txt_tokens)

        # 获取扫描的点云数据、对象标签、对象颜色和对象ID
        obj_pcds = self.get_scan_pcd_data(scan_id)
        obj_labels = self.scans[scan_id]['inst_labels']
        obj_gmm_colors = self.scans[scan_id]['inst_colors']
        obj_ids = [str(x) for x in range(len(obj_labels))]

        # 如果不保留背景，则筛选出非背景对象
        if not self.keep_background:
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if
                                 obj_label not in ['wall', 'floor', 'ceiling']]
            tgt_obj_idx = selected_obj_idxs.index(tgt_obj_idx)
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_gmm_colors = [obj_gmm_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]

        # 随机旋转对象，如果设置为True
        if self.random_rotate:
            theta_idx = np.random.randint(len(ROTATE_ANGLES))
            theta = ROTATE_ANGLES[theta_idx]
        else:
            theta = 0

        # 获取增强后的对象输入数据
        aug_obj_fts, aug_obj_locs, aug_obj_gmm_colors, aug_obj_labels, \
            aug_obj_ids, aug_tgt_obj_idx = self._get_obj_inputs(
            obj_pcds, obj_gmm_colors, obj_labels, obj_ids, tgt_obj_idx,
            theta=theta
        )

        # 将对象特征、位置和颜色数据转换为张量
        aug_obj_fts = torch.from_numpy(aug_obj_fts)
        aug_obj_locs = torch.from_numpy(aug_obj_locs)
        aug_obj_gmm_colors = torch.from_numpy(aug_obj_gmm_colors)
        # 将对象类别转换为对应的整数标签
        aug_obj_classes = torch.LongTensor([self.cat2int[x] for x in aug_obj_labels])

        # 构建输出字典
        outs = {
            'item_ids': item['item_id'],
            'scan_ids': scan_id,
            'txt_ids': txt_tokens,
            'txt_lens': txt_lens,
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


def gtpcd_collate_fn(data):
    '''在数据加载时对数据进行填充和整理。它处理不同长度的文本和对象特征，将它们填充到固定长度，以便可以批量处理'''
    outs = {}
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]

    outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
    outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
    outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])

    outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'], pad_ori_data=True)
    outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0)
    outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0)
    outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
    outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])

    outs['obj_classes'] = pad_sequence(
        outs['obj_classes'], batch_first=True, padding_value=-100
    )
    outs['tgt_obj_idxs'] = torch.LongTensor(outs['tgt_obj_idxs'])
    outs['tgt_obj_classes'] = torch.LongTensor(outs['tgt_obj_classes'])
    return outs
