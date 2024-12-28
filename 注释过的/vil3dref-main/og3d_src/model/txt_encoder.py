'''定义了一个用于文本编码的GloveGRUEncoder类，它使用预训练的GloVe词向量和GRU（门控循环单元）网络来处理文本数据'''
import os
import json
from unicodedata import bidirectional
import jsonlines
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn


# GloveGRUEncoder类，用于文本编码
class GloveGRUEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 加载预处理的文本数据目录
        anno_dir = '/home/shichen/scratch/datasets/referit3d/annotations/glove_tokenized'
        # 加载GloVe词向量
        word_embeds = torch.from_numpy(
            np.load(os.path.join(anno_dir, 'nr3d_vocab_embeds.npy'))
        )
        # 将词向量注册为模型的缓冲区，这样它们就不会被视为模型参数
        self.register_buffer('word_embeds', word_embeds)
        # 初始化GRU网络
        self.gru = nn.GRU(
            input_size=300, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=False, dropout=0
        )

    # 前向传播方法，用于文本编码
    def forward(self, txt_ids, txt_masks):
        # 根据txt_ids获取对应的词向量
        txt_embeds = self.word_embeds[txt_ids]
        # 初始化GRU的隐状态
        h_0 = torch.zeros((self.num_layers, txt_ids.size(0), self.hidden_size)).to(txt_ids.device)
        # 通过GRU网络处理文本
        txt_embeds, _ = self.gru(txt_embeds, h_0)
        # 返回最后一层的隐藏状态
        return EasyDict({
            'last_hidden_state': txt_embeds,
        })


# 准备GloVe分词数据的函数
def prepare_glove_tokenized_data():
    import collections

    # 数据目录
    anno_dir = '/home/shichen/scratch/datasets/referit3d/annotations'
    dataset = 'nr3d'
    outdir = os.path.join(anno_dir, 'glove_tokenized')
    os.makedirs(outdir, exist_ok=True)

    # 加载数据
    data = []
    vocab = collections.Counter()
    with jsonlines.open(os.path.join(anno_dir, 'bert_tokenized', '%s.jsonl' % dataset), 'r') as f:
        for x in f:
            data.append(x)
            for w in x['tokens']:
                vocab[w] += 1
    print(len(vocab))

    # 加载GloVe词向量
    word2vec = {}
    with open('/home/shichen/scratch/datasets/pretrained/wordvecs/glove.42B.300d.txt', 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] in vocab:
                word2vec[tokens[0]] = np.array([float(x) for x in tokens[1:]], dtype=np.float32)

    # 创建词汇表
    int2word = ['unk']
    for w, c in vocab.most_common():
        if w in word2vec:
            int2word.append(w)
    print(len(int2word))
    json.dump(int2word, open(os.path.join(outdir, f'{dataset}_vocab.json'), 'w'), indent=2)

    # 将词向量转换为numpy数组并保存
    word_embeds = [np.zeros(300, dtype=np.float32)]
    for w in int2word[1:]:
        word_embeds.append(word2vec[w])
    np.save(os.path.join(outdir, f'{dataset}_vocab_embeds.npy'), word_embeds)

    # 创建词汇到整数的映射
    word2int = {w: i for i, w in enumerate(int2word)}
    # 将处理后的数据保存到文件
    with jsonlines.open(os.path.join(outdir, f'{dataset}.jsonl'), 'w') as outf:
        for x in data:
            x['enc_tokens'] = [word2int.get(w, 0) for w in x['tokens']]
            outf.write(x)
    

if __name__ == '__main__':
    # prepare_glove_tokenized_data()
    pass