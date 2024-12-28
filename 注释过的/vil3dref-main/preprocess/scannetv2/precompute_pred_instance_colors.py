'''主要功能是对ReferIt3D数据集中点云的颜色信息进行高斯混合模型（Gaussian Mixture Model, GMM）聚类，并将聚类结果以JSON格式保存下来'''
import os
import json
import glob
import torch
import numpy as np

from sklearn.mixture import GaussianMixture

# 定义3D扫描数据的目录路径
scan_dir = 'datasets/referit3d/exprs_neurips22/instance_segm/pointgroup/scan_data'
# 定义输出目录，用于存储实例ID到GMM颜色的映射
output_dir = os.path.join(scan_dir, 'instance_id_to_gmm_color')
# 如果输出目录不存在，则创建它
os.makedirs(output_dir, exist_ok=True)

# 遍历指定目录中所有匹配模式的文件
for scan_file in glob.glob(os.path.join(scan_dir, 'pcd_with_global_alignment', '*')):
    # 从文件名中提取扫描ID
    scan_id = os.path.basename(scan_file).split('.')[0]
    print(scan_file)

    # 加载扫描数据，包括xyz坐标、rgb颜色、语义标签和实例标签
    data = torch.load(scan_file)
    colors = data[1]
    instance_labels = data[3]

    # 如果实例标签为空，则跳过当前迭代
    if instance_labels is None:
        continue

    # 归一化颜色值
    colors = colors / 127.5 - 1

    # 存储每个实例的颜色聚类结果
    clustered_colors = []
    # 遍历每个实例标签
    for i in range(instance_labels.max() + 1):
        # 获取当前实例的掩码
        mask = instance_labels == i
        # 提取当前实例的颜色
        obj_colors = colors[mask]

        # 使用高斯混合模型对当前实例的颜色进行聚类
        gm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(obj_colors)
        # 将聚类结果保存到列表中
        clustered_colors.append({
            'weights': gm.weights_.tolist(),
            'means': gm.means_.tolist(),
        })

    # 将聚类结果保存到JSON文件中
    json.dump(
        clustered_colors,
        open(os.path.join(output_dir, '%s.json' % scan_id), 'w')
    )
