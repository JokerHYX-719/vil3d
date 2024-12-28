'''从点云（PCD）数据中提取每个实例物体的边界框（bounding box），并将边界框信息保存为 .npy 文件'''
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch

def main():
    """
    主函数，用于处理点云数据并生成bounding box信息。
    """
    # 创建ArgumentParser对象，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加点云数据目录的命令行参数
    parser.add_argument('pcd_data_dir')
    # 添加bounding box输出目录的命令行参数
    parser.add_argument('bbox_out_dir')
    # 解析命令行参数
    args = parser.parse_args()

    # 确保bounding box输出目录存在，如果不存在则创建
    os.makedirs(args.bbox_out_dir, exist_ok=True)

    # 获取点云数据目录下的所有文件名，并提取scan id
    scan_ids = [x.split('.')[0] for x in os.listdir(args.pcd_data_dir)]
    # 对scan id进行排序
    scan_ids.sort()

    # 遍历每个scan id，处理点云数据并生成bounding box信息
    for scan_id in tqdm(scan_ids):
        # 加载点云数据，包括点的位置、颜色和实例标签
        points, colors, _, inst_labels = torch.load(
            os.path.join(args.pcd_data_dir, '%s.pth'%scan_id)
        )
        # 如果实例标签为空，则跳过当前scan id
        if inst_labels is None:
            continue
        # 计算实例数量
        num_insts = inst_labels.max()
        # 初始化bounding box信息列表
        outs = []
        # 遍历每个实例，计算bounding box信息
        for i in range(num_insts+1):
            # 获取当前实例的掩码
            inst_mask = inst_labels == i
            # 获取当前实例的点云数据
            inst_points = points[inst_mask]
            # 如果当前实例的点云数据为空，则打印信息并添加空的bounding box信息
            if len(inst_points) == 0:
                print(scan_id, i, 'empty bbox')
                outs.append(np.zeros(6, ).astype(np.float32))
            else:
                # 计算当前实例的bounding box中心点
                bbox_center = inst_points.mean(0)
                # 计算当前实例的bounding box尺寸
                bbox_size = inst_points.max(0) - inst_points.min(0)
                # 将中心点和尺寸信息合并为一个数组，并添加到bounding box信息列表中
                outs.append(np.concatenate([bbox_center, bbox_size], 0))
        # 将所有实例的bounding box信息堆叠为一个数组，并保存为浮点类型
        outs = np.stack(outs, 0).astype(np.float32)

        # 将bounding box信息保存到文件中
        np.save(os.path.join(args.bbox_out_dir, '%s.npy'%scan_id), outs)

# 如果当前脚本为入口点，则调用主函数
if __name__ == '__main__':
    main()

