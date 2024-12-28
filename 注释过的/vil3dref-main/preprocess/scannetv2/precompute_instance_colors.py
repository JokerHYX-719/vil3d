'''
主要功能是对点云数据中每个实例的颜色信息进行聚类，并将聚类结果保存为 .json 文件。
具体来说，它从点云数据中提取每个实例的颜色数据，使用高斯混合模型（Gaussian Mixture Model, GMM）对颜色进行聚类，然后将每个实例的颜色聚类结果（包括聚类权重和均值）保存下来。主要步骤如下：
'''
import os
import json
import glob
import torch
import numpy as np

from sklearn.mixture import GaussianMixture

# 定义数据集目录和输出目录
scan_dir = 'datasets/referit3d/scan_data'
output_dir = os.path.join(scan_dir, 'instance_id_to_gmm_color')
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历所有扫描文件
for scan_file in glob.glob(os.path.join(scan_dir, 'pcd_with_global_alignment', '*')):
    # 提取scan_id
    scan_id = os.path.basename(scan_file).split('.')[0]
    print(scan_file)

    # 加载扫描数据，包括颜色和实例标签等信息
    data = torch.load(scan_file) # xyz, rgb, semantic_labels, instance_labels
    colors = data[1]
    instance_labels = data[3]

    # 如果实例标签为空，则跳过当前文件
    if instance_labels is None:
        continue

    # 颜色归一化
    colors = colors / 127.5 - 1

    clustered_colors = []
    # 对每个实例进行颜色聚类
    for i in range(instance_labels.max() + 1):
        # 创建当前实例的掩码
        mask = instance_labels == i     # time consuming
        obj_colors = colors[mask]

        # 使用GaussianMixture模型进行颜色聚类
        gm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(obj_colors)
        # 保存聚类结果的权重和均值
        clustered_colors.append({
            'weights': gm.weights_.tolist(),
            'means': gm.means_.tolist(),
        })

    # 将聚类结果保存到json文件中
    json.dump(
        clustered_colors,
        open(os.path.join(output_dir, '%s.json'%scan_id), 'w')
    )
