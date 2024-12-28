import os
import sys
import json
import numpy as np
import time
from collections import defaultdict
from tqdm import tqdm
from easydict import EasyDict
import pprint

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.logger import LOGGER, TB_LOGGER, AverageMeter, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched_decay_rate
from optim.misc import build_optimizer

from parser import load_parser, parse_with_config

from data.gtlabelpcd_dataset import GTLabelPcdDataset, gtlabelpcd_collate_fn

from model.referit3d_net_mix import ReferIt3DNetMix


def build_datasets(data_cfg):
    trn_dataset = GTLabelPcdDataset(
        data_cfg.trn_scan_split, data_cfg.anno_file,
        data_cfg.scan_dir, data_cfg.category_file,
        cat2vec_file=data_cfg.cat2vec_file,
        random_rotate=data_cfg.get('random_rotate', False),
        max_txt_len=data_cfg.max_txt_len, max_obj_len=data_cfg.max_obj_len,
        keep_background=data_cfg.get('keep_background', False),
        num_points=data_cfg.num_points, in_memory=True,
        gt_scan_dir=data_cfg.get('gt_scan_dir', None),
        iou_replace_gt=data_cfg.get('iou_replace_gt', 0),
    )
    val_dataset = GTLabelPcdDataset(
        data_cfg.val_scan_split, data_cfg.anno_file,
        data_cfg.scan_dir, data_cfg.category_file,
        cat2vec_file=data_cfg.cat2vec_file,
        max_txt_len=None, max_obj_len=None, random_rotate=False,
        keep_background=data_cfg.get('keep_background', False),
        num_points=data_cfg.num_points, in_memory=True,
        gt_scan_dir=data_cfg.get('gt_scan_dir', None),
        iou_replace_gt=data_cfg.get('iou_replace_gt', 0),
    )
    return trn_dataset, val_dataset


def main(opts):
    # 设置CUDA环境，确定是否使用默认GPU，GPU数量以及设备对象
    default_gpu, n_gpu, device = set_cuda(opts)

    # 如果使用默认GPU，记录设备信息和分布式训练状态
    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1)
            )
        )

    # 设置随机种子以保证实验可重复性
    seed = opts.seed
    if opts.local_rank != -1:
        seed += opts.rank
    set_random_seed(seed)

    # 如果使用默认GPU，保存训练元数据，创建TensorBoard日志和模型保存器
    if default_gpu:
        if not opts.test:
            save_training_meta(opts)
            TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
            model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
            add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    else:
        # 如果不使用默认GPU，禁用日志记录和设置空操作
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # 准备模型，创建ReferIt3DNetMix模型实例
    model_config = EasyDict(opts.model)
    model = ReferIt3DNetMix(model_config, device)

    # 计算模型参数数量和可训练参数数量
    num_weights, num_trainable_weights = 0, 0
    for p in model.parameters():
        psize = np.prod(p.size())
        num_weights += psize
        if p.requires_grad:
            num_trainable_weights += psize
    LOGGER.info('#weights: %d, #trainable weights: %d', num_weights, num_trainable_weights)

    # 如果提供了恢复文件，加载模型参数
    if opts.resume_files:
        checkpoint = {}
        for resume_file in opts.resume_files:
            new_checkpoints = torch.load(resume_file, map_location=lambda storage, loc: storage)
            for k, v in new_checkpoints.items():
                if k not in checkpoint:
                    checkpoint[k] = v
        if len(opts.resume_files) == 1:
            model.load_state_dict(checkpoint)
        print(
            'resume #params:', len(checkpoint),
            len([n for n in checkpoint.keys() if n in model.teacher_model.state_dict()]),
            len([n for n in checkpoint.keys() if n in model.student_model.state_dict()]),
        )

        model.teacher_model.load_state_dict(checkpoint, strict=False)
        if opts.resume_student:
            model.student_model.load_state_dict(checkpoint, strict=False)
        else:
            student_checkpoint = torch.load(
                opts.resume_files[0], map_location=lambda storage, loc: storage
            )
            print('resume_student', len(student_checkpoint))
            model.student_model.load_state_dict(student_checkpoint, strict=False)

    # 根据模型配置，包装模型以支持分布式训练
    model_cfg = model.config
    model = wrap_model(model, device, opts.local_rank)

    # 加载训练和验证数据集
    data_cfg = EasyDict(opts.dataset)
    trn_dataset, val_dataset = build_datasets(data_cfg)
    collate_fn = gtlabelpcd_collate_fn
    LOGGER.info('train #scans %d, #data %d' % (len(trn_dataset.scan_ids), len(trn_dataset)))
    LOGGER.info('val #scans %d, #data %d' % (len(val_dataset.scan_ids), len(val_dataset)))

    # 构建数据加载器
    if opts.local_rank == -1:
        trn_sampler = None
        pre_epoch = lambda e: None
    else:
        size = dist.get_world_size()
        trn_sampler = DistributedSampler(
            trn_dataset, num_replicas=size, rank=dist.get_rank(), shuffle=True
        )
        pre_epoch = trn_sampler.set_epoch

    trn_dataloader = DataLoader(
        trn_dataset, batch_size=opts.batch_size, shuffle=True if trn_sampler is None else False,
        num_workers=opts.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=1,
        sampler=trn_sampler
    )  # DataLoader 中的一个参数，用于控制数据预取的数量
    val_dataloader = DataLoader(
        val_dataset, batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=1,
    )
    # 计算训练步数
    opts.num_train_steps = len(trn_dataloader) * opts.num_epoch

    # 如果是测试模式，执行验证并保存预测结果
    if opts.test:
        val_log, out_preds = validate(model, model_cfg, val_dataloader, return_preds=True)
        pred_dir = os.path.join(opts.output_dir, 'preds')
        os.makedirs(pred_dir, exist_ok=True)
        json.dump(out_preds, open(os.path.join(pred_dir, 'val_outs.json'), 'w'))
        return

    # 准备优化器
    optimizer, init_lrs = build_optimizer(model, opts)

    # 记录训练信息
    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.batch_size if opts.local_rank == -1 else opts.batch_size * opts.world_size)
    LOGGER.info("  Num epoch = %d, num steps = %d", opts.num_epoch, opts.num_train_steps)

    # 用于计算训练统计信息
    avg_metrics = defaultdict(AverageMeter)

    global_step = 0

    # 设置模型为训练模式
    model.train()
    optimizer.zero_grad()
    optimizer.step()

    # 如果使用默认GPU，执行一次验证
    if default_gpu:
        val_log = validate(model, model_cfg, val_dataloader)

    # 初始化最佳验证分数
    val_best_scores = {'epoch': -1, 'acc/og3d': -float('inf')}
    epoch_iter = range(opts.num_epoch)
    # 如果使用默认GPU，显示进度条
    if default_gpu:
        epoch_iter = tqdm(epoch_iter)
    for epoch in epoch_iter:
        pre_epoch(epoch)  # 为分布式训练设置epoch

        start_time = time.time()
        # 如果使用默认GPU，显示批次进度条
        batch_iter = trn_dataloader
        if default_gpu:
            batch_iter = tqdm(batch_iter)
        for batch in batch_iter:
            batch_size = len(batch['scan_ids'])
            # 执行模型前向传播和损失计算
            result, losses = model(batch, compute_loss=True)
            # 反向传播和优化器更新
            losses['total'].backward()

            # 优化器更新和记录
            global_step += 1
            # 学习率调度
            lr_decay_rate = get_lr_sched_decay_rate(global_step, opts)
            for kp, param_group in enumerate(optimizer.param_groups):  # 根据初始学习率和学习率衰减率计算当前步骤的学习率
                param_group['lr'] = lr_this_step = init_lrs[kp] * lr_decay_rate
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # 记录损失
            # 注意：这里没有跨GPU收集数据，以提高效率
            loss_dict = {'loss/%s' % lk: lv.data.item() for lk, lv in losses.items()}
            for lk, lv in loss_dict.items():
                avg_metrics[lk].update(lv, n=batch_size)
            TB_LOGGER.log_scalar_dict(loss_dict)
            TB_LOGGER.step()

            # 更新模型参数
            if opts.grad_norm != -1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opts.grad_norm
                )  # 如果启用了梯度裁剪，调用 torch.nn.utils.clip_grad_norm_ 并记录裁剪后的梯度范数
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
            # break

        LOGGER.info(
            'Epoch %d, lr: %.6f, %s', epoch + 1,
            optimizer.param_groups[-1]['lr'],
            ', '.join(['%s: %.4f' % (lk, lv.avg) for lk, lv in avg_metrics.items()])
        )
        # 如果使用默认GPU并且到了验证周期，执行验证
        if default_gpu and (epoch + 1) % opts.val_every_epoch == 0:
            LOGGER.info(f'------Epoch {epoch + 1}: start validation (val)------')
            val_log = validate(model, model_cfg, val_dataloader)
            TB_LOGGER.log_scalar_dict(
                {f'valid/{k}': v.avg for k, v in val_log.items()}
            )
            # 保存模型
            output_model_file = model_saver.save(
                model, epoch + 1, optimizer=optimizer, save_latest_optim=True
            )
            if val_log['acc/og3d'].avg > val_best_scores['acc/og3d']:
                val_best_scores['acc/og3d'] = val_log['acc/og3d'].avg
                val_best_scores['epoch'] = epoch + 1
                model_saver.remove_previous_models(epoch + 1)
            else:
                os.remove(output_model_file)

    LOGGER.info('Finished training!')
    LOGGER.info(
        'best epoch: %d, best acc/og3d %.4f', val_best_scores['epoch'], val_best_scores['acc/og3d']
    )


@torch.no_grad()
def validate(model, model_cfg, val_dataloader, niters=None, return_preds=False):
    # 在不计算梯度的情况下进行验证
    model.eval()  # 将模型设置为评估模式

    avg_metrics = defaultdict(AverageMeter)  # 用于存储平均的评估指标
    out_preds = {}  # 存储预测输出的字典
    for ib, batch in enumerate(val_dataloader):  # 遍历验证数据集的每个 batch
        batch_size = len(batch['scan_ids'])  # 获取当前 batch 的大小
        # 通过模型得到结果和损失值
        result, losses = model(batch, compute_loss=True, is_test=True)

        # 计算每个损失值并更新平均损失
        loss_dict = {'loss/%s' % lk: lv.data.item() for lk, lv in losses.items()}
        for lk, lv in loss_dict.items():
            avg_metrics[lk].update(lv, n=batch_size)

        # 对 3D 物体进行分类，获取预测结果，并计算分类精度
        og3d_preds = torch.argmax(result['og3d_logits'], dim=1).cpu()  # 选择预测概率最大的类别
        avg_metrics['acc/og3d'].update(
            torch.mean((og3d_preds == batch['tgt_obj_idxs']).float()).item(),  # 计算物体分类精度
            n=batch_size
        )
        # 计算 3D 物体类别的精度
        avg_metrics['acc/og3d_class'].update(
            torch.mean((batch['obj_classes'].gather(1, og3d_preds.unsqueeze(1)).squeeze(1) == batch[
                # unsqueeze添加一个维度，以匹配 gather 操作所需的维度。squeeze(1) 移除添加的维度，得到一个预测类别的张量
                'tgt_obj_classes']).float()).item(),
            n=batch_size
        )

        # 如果配置启用了 3D 物体分类损失，则计算相关精度
        if model_cfg.losses.obj3d_clf:
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_logits'], dim=2).cpu()
            avg_metrics['acc/obj3d_clf'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )
        # 如果启用了预训练的 3D 物体分类损失，计算精度
        if model_cfg.losses.obj3d_clf_pre:
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_pre_logits'], dim=2).cpu()
            avg_metrics['acc/obj3d_clf_pre'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )
        # 如果启用了文本分类损失，计算文本分类精度
        if model_cfg.losses.txt_clf:
            txt_clf_preds = torch.argmax(result['txt_clf_logits'], dim=1).cpu()
            avg_metrics['acc/txt_clf'].update(
                (txt_clf_preds == batch['tgt_obj_classes']).float().mean().item(),
                n=batch_size
            )
        # 如果启用了旋转分类损失，计算每层旋转分类的精度
        if model_cfg.losses.get('rot_clf', False):
            for il in model_cfg.mm_encoder.rot_layers:  # 遍历多模态编码器中的旋转层
                rot_clf_preds = result['all_rot_preds'][il].cpu()  # 获取旋转预测结果
                gt_views = batch['target_views']  # 获取目标视角
                gt_view_mask = gt_views != -100  # 创建视角的有效掩码
                avg_metrics['acc/rot_clf_%d' % il].update(
                    (rot_clf_preds == gt_views)[gt_view_mask].float().mean().item(),  # 计算精度
                    n=torch.sum(gt_view_mask)
                )
        # 如果启用了文本对比损失，计算文本对比任务的精度
        if model_cfg.losses.get('txt_contrast', False):
            txt_ctr_preds = result['txt_pos_sims'] > torch.max(result['txt_neg_sims'], 1)[0]  # 对比正负样本的相似度
            txt_ctr_preds = txt_ctr_preds.cpu().float()  # 转换为浮点数
            avg_metrics['acc/txt_contrast'].update(
                txt_ctr_preds.mean().item(),  # 计算对比精度
                n=batch_size
            )

        # 如果需要返回预测结果，则保存每个样本的预测信息
        if return_preds:
            for ib in range(batch_size):
                out_preds[batch['item_ids'][ib]] = {
                    'obj_ids': batch['obj_ids'][ib],
                    'obj_logits': result['og3d_logits'][ib, :batch['obj_lens'][ib]].data.cpu().numpy().tolist(),
                }
        # 如果迭代次数设置了限制且已达到，则终止验证
        if niters is not None and ib >= niters:
            break

    # 打印每个指标的平均值
    LOGGER.info(', '.join(['%s: %.4f' % (lk, lv.avg) for lk, lv in avg_metrics.items()]))

    model.train()  # 恢复模型为训练模式
    # 返回预测结果或仅返回平均指标
    if return_preds:
        return avg_metrics, out_preds
    return avg_metrics


def build_args():
    parser = load_parser()
    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )

    return opts


if __name__ == '__main__':
    args = build_args()
    pprint.pprint(args)
    main(args)
