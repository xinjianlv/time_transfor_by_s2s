import os , pdb
from tqdm import tqdm

import json
from collections import Counter
from numpy import *
import torch
from torch.utils.data import DataLoader, TensorDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_vocab(texts, n=None):
    counter = Counter(''.join(texts))  # char level
    char2index = {w: i for i, (w, c) in enumerate(counter.most_common(n), start=4)}
    char2index['~'] = 0  # pad  不足长度的文本在后边填充0
    char2index['^'] = 1  # sos  表示句子的开头
    char2index['$'] = 2  # eos  表示句子的结尾
    char2index['#'] = 3  # unk  表示句子中出现的字典中没有的未知词
    index2char = {i: w for w, i in char2index.items()}
    return char2index, index2char


def indexes_from_text(text, char2index):
    return [1] + [char2index[c] for c in text] + [2]  # 手动添加开始结束标志
def pad_seq(seq, max_length):
    # seq += [0 for _ in range(max_length - len(seq))]
    seq += [0] * (max_length - len(seq))
    return seq

def get_data_loaders(data_file , batch_size, train_precent):

    pairs = json.load(open(data_file, 'rt', encoding='utf-8'))
    data = array(pairs)
    src_texts = data[:, 0]
    trg_texts = data[:, 1]
    src_c2ix, src_ix2c = build_vocab(src_texts)
    trg_c2ix, trg_ix2c = build_vocab(trg_texts)

    max_src_len = max(list(map(len, src_texts))) + 2
    max_trg_len = max(list(map(len, trg_texts))) + 2
    max_src_len, max_trg_len


    input_seqs, target_seqs = [], []
    for i in range(len(pairs)):
        input_seqs.append(indexes_from_text(pairs[i][0], src_c2ix))
        target_seqs.append(indexes_from_text(pairs[i][1], trg_c2ix))

    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]


    input_var = torch.LongTensor(input_padded)
    target_var = torch.LongTensor(target_padded)

    tensor_datasets = {'train': [], 'valid': []}
    data_set = {'input_ids': input_var, 'input_length':input_lengths, 'target_ids': target_var}
    train_max_num = int(len(data_set['input_ids']) * train_precent)

    for name  in ['input_ids' ,'input_length' ,'target_ids']: ##['input_ids' , 'label_ids']
        tensor_datasets['train'].append(torch.LongTensor(data_set[name][0:train_max_num]))
        tensor_datasets['valid'].append(torch.LongTensor(data_set[name][train_max_num:]))

    train_data_set, valid_data_set = TensorDataset(*tensor_datasets['train']), TensorDataset(*tensor_datasets['valid'])
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data_set, batch_size=batch_size)

    return train_data_loader, valid_data_loader, len(src_c2ix), len(trg_c2ix)


if __name__ == '__main__':

    data_file = '../data/time_transfor/Time Dataset.json'
    batch_size = 128
    train_precent = 0.7

    train_data_loader, valid_data_loader, input_lengths, target_lengths = get_data_loaders(data_file , batch_size , train_precent)

    print('end...')
