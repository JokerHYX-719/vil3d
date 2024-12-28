import os
import jsonlines
import json
import numpy as np
import random
import collections
import copy

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

try:    
    from .common import pad_tensors, gen_seq_masks
except:
    from common import pad_tensors, gen_seq_masks


ROTATE_ANGLES = [0, np.pi/2, np.pi, np.pi*3/2]


class GTLabelDataset(Dataset):
    '''它用于加载和处理与目标标签相关的数据。这个数据集类专为处理具有文本描述和3D对象标注信息的数据而设计'''
    def __init__(
        self, scan_id_file, anno_file, scan_dir, category_file,
        cat2vec_file=None, keep_background=False, random_rotate=False, 
        max_txt_len=50, max_obj_len=80, gt_scan_dir=None, iou_replace_gt=0
    ):
        '''
        scan_id_file：包含扫描ID的文件路径，标识了数据集中包含哪些扫描。
        anno_file：注释文件的路径，包含了文本描述和其他元数据。
        scan_dir：扫描数据的目录路径。
        category_file：类别文件的路径，包含了类别到索引的映射。
        cat2vec_file：可选的类别到向量的映射文件路径。
        keep_background：是否保留背景对象。
        random_rotate：是否对数据进行随机旋转增强。
        max_txt_len 和 max_obj_len：文本和对象标注的最大长度。
        gt_scan_dir：可选的，用于加载地面真实扫描数据的目录路径。
        iou_replace_gt：用于确定是否用预测的标注替换地面真实的标注，基于IOU（交并比）
        '''
        super().__init__()

        ## 初始化数据集，加载扫描ID、注释文件、扫描目录、类别文件等信息
        split_scan_ids = set([x.strip() for x in open(scan_id_file, 'r')])

        self.scan_dir = scan_dir
        self.max_txt_len = max_txt_len
        self.max_obj_len = max_obj_len
        self.keep_background = keep_background
        self.random_rotate = random_rotate
        self.gt_scan_dir = gt_scan_dir
        self.iou_replace_gt = iou_replace_gt

        self.scan_ids = set()
        self.data = []
        self.scan_to_item_idxs = collections.defaultdict(list)
        with jsonlines.open(anno_file, 'r') as f:
            for item in f:
                if item['scan_id'] in split_scan_ids:
                    if (len(item['tokens']) > 24) and (not item['item_id'].startswith('scanrefer')): continue
                    # if not is_explicitly_view_dependent(item['tokens']): continue
                    self.scan_ids.add(item['scan_id'])
                    self.scan_to_item_idxs[item['scan_id']].append(len(self.data))
                    self.data.append(item)

        ## 加载每个扫描的实例标签、位置和颜色信息
        # 初始化一个字典来存储每个扫描的标签、位置和颜色信息
        self.scans = {}
        # 遍历所有的扫描ID，加载每个扫描的实例标签、位置和颜色信息
        for scan_id in self.scan_ids:
            # 加载实例标签，将实例ID映射到名称
            inst_labels = json.load(open(os.path.join(scan_dir, 'instance_id_to_name', '%s.json' % scan_id)))
            # 加载实例位置，包括中心点坐标和尺寸信息
            inst_locs = np.load(os.path.join(scan_dir, 'instance_id_to_loc', '%s.npy' % scan_id))
            # 加载实例颜色，使用高斯混合模型（GMM）表示颜色分布
            inst_colors = json.load(open(os.path.join(scan_dir, 'instance_id_to_gmm_color', '%s.json' % scan_id)))
            # 将颜色信息转换为包含权重和均值的数组
            inst_colors = [np.concatenate(
                [np.array(x['weights'])[:, None], np.array(x['means'])],
                axis=1
            ).astype(np.float32) for x in inst_colors]
            # 将加载的信息存储到字典中，与扫描ID关联
            self.scans[scan_id] = {
                'inst_labels': inst_labels,  # (n_obj, )
                'inst_locs': inst_locs,  # (n_obj, 6) center xyz, whl
                'inst_colors': inst_colors,  # (n_obj, 3x4) cluster * (weight, mean rgb)
            }
        # 如果提供了地面真值扫描目录，则加载地面真值数据
        if self.gt_scan_dir is not None:
            for scan_id in self.scan_ids:
                # 类似于上述过程，但加载的是地面真值数据
                inst_labels = json.load(open(os.path.join(gt_scan_dir, 'instance_id_to_name', '%s.json' % scan_id)))
                inst_locs = np.load(os.path.join(gt_scan_dir, 'instance_id_to_loc', '%s.npy' % scan_id))
                inst_colors = json.load(
                    open(os.path.join(gt_scan_dir, 'instance_id_to_gmm_color', '%s.json' % scan_id)))
                inst_colors = [np.concatenate(
                    [np.array(x['weights'])[:, None], np.array(x['means'])],
                    axis=1
                ).astype(np.float32) for x in inst_colors]
                # 更新字典，添加地面真值数据
                self.scans[scan_id].update({
                    'gt_inst_labels': inst_labels,  # (n_obj, )
                    'gt_inst_locs': inst_locs,  # (n_obj, 6) center xyz, whl
                    'gt_inst_colors': inst_colors,  # (n_obj, 3x4) cluster * (weight, mean rgb)
                })

        #加载类别到向量的映射
        self.int2cat = json.load(open(category_file, 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        if cat2vec_file is None:
            self.cat2vec = None
        else:
            self.cat2vec = json.load(open(cat2vec_file, 'r'))
        
    def __len__(self):
        # 返回数据集中的数据点数量
        return len(self.data)

    def _get_obj_inputs(self, obj_labels, obj_locs, obj_colors, obj_ids, tgt_obj_idx, theta=None):
        """
        处理对象输入，包括旋转和选择对象。

        参数:
            obj_labels (list): 对象的标签列表。
            obj_locs (list): 对象的位置列表。
            obj_colors (list): 对象的颜色列表。
            obj_ids (list): 对象的ID列表。
            tgt_obj_idx (int): 目标对象的索引。
            theta (float, optional): 旋转角度。如果为None，则不进行旋转。

        返回:
            obj_labels (list): 处理后的对象标签列表。
            obj_locs (numpy.array): 处理后的对象位置数组。
            obj_colors (numpy.array): 处理后的对象颜色数组。
            obj_ids (list): 处理后的对象ID列表。
            tgt_obj_idx (int): 处理后的目标对象索引。
        """
        tgt_obj_type = obj_labels[tgt_obj_idx]
        # 如果对象数量超过最大长度，则选择与目标对象相同类型的其他对象
        if (self.max_obj_len is not None) and (len(obj_labels) > self.max_obj_len):
            selected_obj_idxs = [tgt_obj_idx]
            remained_obj_idxs = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj != tgt_obj_idx:
                    if klabel == tgt_obj_type:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idxs.append(kobj)
            # 如果选择的对象数量不足最大长度，则随机选择剩余对象补充
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idxs)
                selected_obj_idxs += remained_obj_idxs[:self.max_obj_len - len(selected_obj_idxs)]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_locs = [obj_locs[i] for i in selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]
            tgt_obj_idx = 0

        # 将对象位置和颜色转换为numpy数组
        obj_locs = np.array(obj_locs)
        obj_colors = np.array(obj_colors)

        # 如果指定了旋转角度，则对对象位置进行旋转
        if (theta is not None) and (theta != 0):
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            obj_locs[:, :3] = np.matmul(obj_locs[:, :3], rot_matrix.transpose())

        return obj_labels, obj_locs, obj_colors, obj_ids, tgt_obj_idx

    def __getitem__(self, idx):
        """
        获取单个数据样本。

        参数:
            idx (int): 数据样本的索引。

        返回:
            outs (dict): 包含处理后的数据样本的字典。
        """
        item = self.data[idx]
        scan_id = item['scan_id']
        tgt_obj_idx = item['target_id']
        tgt_obj_type = item['instance_type']

        # 加载文本数据
        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        txt_lens = len(txt_tokens)

        # 加载对象数据
        if self.gt_scan_dir is None or item['max_iou'] > self.iou_replace_gt:
            obj_labels = self.scans[scan_id]['inst_labels']
            obj_locs = self.scans[scan_id]['inst_locs']
            obj_colors = self.scans[scan_id]['inst_colors']
        else:
            tgt_obj_idx = item['gt_target_id']
            obj_labels = self.scans[scan_id]['gt_inst_labels']
            obj_locs = self.scans[scan_id]['gt_inst_locs']
            obj_colors = self.scans[scan_id]['gt_inst_colors']

        obj_ids = [str(x) for x in range(len(obj_labels))]

        # 如果不保留背景，则过滤背景对象
        if not self.keep_background:
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if
                                 obj_label not in ['wall', 'floor', 'ceiling']]
            tgt_obj_idx = selected_obj_idxs.index(tgt_obj_idx)
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_locs = obj_locs[selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]

        # 如果启用随机旋转，则随机选择一个旋转角度
        if self.random_rotate:
            theta_idx = np.random.randint(len(ROTATE_ANGLES))
            theta = ROTATE_ANGLES[theta_idx]
        else:
            theta = 0
        # 处理对象输入
        aug_obj_labels, aug_obj_locs, aug_obj_colors, aug_obj_ids, aug_tgt_obj_idx = \
            self._get_obj_inputs(
                obj_labels, obj_locs, obj_colors, obj_ids, tgt_obj_idx,
                theta=theta
            )

        # 将对象位置和颜色转换为PyTorch张量
        aug_obj_locs = torch.from_numpy(aug_obj_locs)
        aug_obj_colors = torch.from_numpy(aug_obj_colors)
        aug_obj_classes = torch.LongTensor([self.cat2int[x] for x in aug_obj_labels])
        # 如果提供了类别向量，则使用类别向量；否则使用类别索引
        if self.cat2vec is None:
            aug_obj_fts = aug_obj_classes
        else:
            aug_obj_fts = torch.FloatTensor([self.cat2vec[x] for x in aug_obj_labels])

        # 构建输出字典
        outs = {
            'item_ids': item['item_id'],
            'scan_ids': scan_id,
            'txt_ids': txt_tokens,
            'txt_lens': txt_lens,
            'obj_fts': aug_obj_fts,
            'obj_locs': aug_obj_locs,
            'obj_colors': aug_obj_colors,
            'obj_lens': len(aug_obj_fts),
            'obj_classes': aug_obj_classes,
            'tgt_obj_idxs': aug_tgt_obj_idx,
            'tgt_obj_classes': self.cat2int[tgt_obj_type],
            'obj_ids': aug_obj_ids,
        }

        return outs

def gtlabel_collate_fn(data):
    '''收集函数，用于在数据加载时对数据进行填充和整理'''
    outs = {}
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]

    # 对文本数据进行填充
    outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
    outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
    outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])

    # 对对象特征进行填充
    if len(outs['obj_fts'][0].size()) == 1:
        outs['obj_fts'] = pad_sequence(outs['obj_fts'], batch_first=True)
    else:
        outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'])
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

