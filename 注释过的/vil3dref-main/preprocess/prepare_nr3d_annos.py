import os
import argparse
import json
import jsonlines
import pandas as pd

from transformers import AutoTokenizer, AutoModel

def main():
    """
    主函数，用于处理输入文件并生成处理后的输出文件。
    该函数使用命令行参数来接收输入文件和输出文件的路径，
    并对输入文件中的数据进行处理，保存到输出文件中。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加输入文件路径参数
    parser.add_argument('--input_file')
    # 添加输出文件路径参数
    parser.add_argument('--output_file')
    # 解析命令行参数
    args = parser.parse_args()

    # 确保输出文件的目录存在，如果不存在则创建
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 加载预训练的BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 读取输入文件，假设输入文件是CSV格式且包含表头
    data = pd.read_csv(args.input_file, header=0)
    # 打印输入数据的项数
    print('process %d items' % (len(data)))

    # 初始化保留项的数量计数器
    num_reserved_items = 0
    # 打开输出文件，使用jsonlines以写入模式
    with jsonlines.open(args.output_file, 'w') as outf:
        # 遍历输入数据的每一项
        for i in range(len(data)):
            item = data.iloc[i]
            # 如果项目不提及目标类别，则跳过
            if not item['mentions_target_class']:
                continue
            # 使用tokenizer编码utterance文本
            enc_tokens = tokenizer.encode(item['utterance'])
            # 构建新的项目字典，包含必要的信息
            new_item = {
                'item_id': '%s_%06d' % (item['dataset'], i),
                'stimulus_id': item['stimulus_id'],
                'scan_id': item['scan_id'],
                'instance_type': item['instance_type'],
                'target_id': int(item['target_id']),
                'utterance': item['utterance'],
                'tokens': eval(item['tokens']),
                'enc_tokens': enc_tokens,
                'correct_guess': bool(item['correct_guess']),
            }
            # 根据数据集类型添加特定的字段
            if item['dataset'] == 'nr3d':
                new_item.update({
                    'uses_object_lang': bool(item['uses_object_lang']),
                    'uses_spatial_lang': bool(item['uses_spatial_lang']),
                    'uses_color_lang': bool(item['uses_color_lang']),
                    'uses_shape_lang': bool(item['uses_shape_lang'])
                })
            else:
                new_item.update({
                    'coarse_reference_type': item['coarse_reference_type'],
                    'reference_type': item['reference_type'],
                    'anchors_types': eval(item['anchors_types']),
                    'anchor_ids': eval(item['anchor_ids']),
                })
            # 写入新的项目到输出文件
            outf.write(new_item)
            # 更新保留项的数量
            num_reserved_items += 1

    # 打印保留的项数
    print('keep %d items' % (num_reserved_items))

if __name__ == '__main__':
    main()