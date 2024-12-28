'''将 ScanRefer 数据集中的文本描述通过 BERT 分词器进行预处理，将文本转化为模型可以理解的编码格式，并将这些编码数据与其他元数据（如场景 ID、物体 ID 等）一起输出到一个 JSON Lines 文件中'''
import os
import argparse
import json
import jsonlines

from transformers import AutoTokenizer, AutoModel

def main():
    '''
    item_id: 数据项的唯一标识符（由 split 和索引 i 组合生成）。
    scan_id: 场景 ID。
    target_id: 目标物体 ID。
    instance_type: 物体类型（去掉下划线）。
    utterance: 自然语言描述。
    tokens: 原始 token 列表。
    enc_tokens: 编码后的 token 列表（由 BERT 的分词器生成）。
    ann_id: 注释 ID。
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    num_reserved_items = 0
    with jsonlines.open(args.output_file, 'w') as outf:
        for split in ['train', 'val', 'test']:
            data = json.load(open(os.path.join(args.input_dir, 'ScanRefer_filtered_%s.json'%split)))
            print('process %s: %d items' % (split, len(data)))
            for i, item in enumerate(data):
                enc_tokens = tokenizer.encode(item['description'])
                outf.write({
                    'item_id': 'scanrefer_%s_%06d' % (split, i),
                    'scan_id': item['scene_id'],
                    'target_id': int(item['object_id']),
                    'instance_type': item['object_name'].replace('_', ' '),
                    'utterance': item['description'],
                    'tokens': item['token'],
                    'enc_tokens': enc_tokens,
                    'ann_id': item['ann_id']
                })
                num_reserved_items += 1

    print('keep %d items' % (num_reserved_items))

if __name__ == '__main__':
    main()