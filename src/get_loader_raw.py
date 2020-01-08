import os , pdb

import json
import torch
import zipfile as zf
from collections import Counter
from numpy import *
from torch.utils.data import DataLoader, TensorDataset
from mylog import logger

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

def build_vocab_raw(texts , min=0):
    char2index = {}
    char2index['<pad>'] = 0  # pad  不足长度的文本在后边填充0
    char2index['<s>'] = 1  # sos  表示句子的开头
    char2index['<e>'] = 2  # eos  表示句子的结尾
    char2index['unk'] = 3  # unk  表示句子中出现的字典中没有的未知词
    text_all = ' '.join(texts)
    text_all_words = text_all.split(' ')
    word_temp_dict = {}
    for word in text_all_words:
        count = 1
        if word in word_temp_dict.keys():
            count += word_temp_dict.get(word)
        word_temp_dict[word] = count
    word_count_sorted = sorted(word_temp_dict.items(), key=lambda d: d[1])
    words_filtered = [n for n in word_count_sorted if n[1] > min]
    for ndx , k_v in enumerate(words_filtered , start=4):
        char2index[k_v[0]] = ndx
    index2char = {i: w for w, i in char2index.items()}
    return char2index , index2char


def indexes_from_text(text, char2index):
    return [1] + [char2index[c] for c in text] + [2]  # 手动添加开始结束标志


def pad_seq(seq, max_length):
    # seq += [0 for _ in range(max_length - len(seq))]
    seq += [0] * (max_length - len(seq))
    return seq


def get_data_loaders(sfile , tfile, batch_size, train_precent, password , distributed=False):
    pairs = load_raw_data(sfile , tfile , password)
    data = array(pairs)
    src_texts = data[:, 0]
    trg_texts = data[:, 1]
    logger.info('build vocab...')
    src_c2ix, src_ix2c = build_vocab_raw(src_texts)
    trg_c2ix, trg_ix2c = build_vocab_raw(trg_texts)

    max_src_len = max(list(map(len, src_texts))) + 2
    max_trg_len = max(list(map(len, trg_texts))) + 2
    max_src_len, max_trg_len

    logger.info('build input and target vectors...')
    input_seqs, target_seqs = [], []
    for i in range(len(pairs)):
        if len(pairs[i][0].split(' ')) > 100 or len(pairs[i][1].split(' ')) > 100:
            continue
        input_seqs.append(indexes_from_text(pairs[i][0].split(' '), src_c2ix))
        target_seqs.append(indexes_from_text(pairs[i][1].split(' '), trg_c2ix))

    logger.info('sort seq_pairs...')
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    logger.info('zip seq_pairs...')
    input_seqs, target_seqs = zip(*seq_pairs)
    logger.info('padding input_seqs')
    input_lengths = [len(s) for s in input_seqs]
    max_length = max(input_lengths)
    logger.info('max input lengths %d'%(max_length))
    input_padded = []
    for inputs_ins in input_seqs:
        input_padded.append(pad_seq(inputs_ins, max_length))
    logger.info('padding target_seqs')
    target_lengths = [len(s) for s in target_seqs]
    max_length = max(target_lengths)
    target_padded = []
    for target_ins in target_seqs:
        target_padded.append(pad_seq(target_ins, max_length))

    input_var = torch.LongTensor(input_padded)
    target_var = torch.LongTensor(target_padded)

    tensor_datasets = {'train': [], 'valid': []}
    data_set = {'input_ids': input_var, 'input_length':input_lengths, 'target_ids': target_var}
    train_max_num = int(len(data_set['input_ids']) * train_precent)

    logger.info('build datasets...')
    for name  in ['input_ids' ,'input_length' ,'target_ids']: ##['input_ids' , 'label_ids']
        tensor_datasets['train'].append(torch.LongTensor(data_set[name][0:train_max_num]))
        tensor_datasets['valid'].append(torch.LongTensor(data_set[name][train_max_num:]))

    train_data_set, valid_data_set = TensorDataset(*tensor_datasets['train']), TensorDataset(*tensor_datasets['valid'])
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle = True, drop_last= True)
    valid_data_loader = DataLoader(valid_data_set, batch_size=batch_size, shuffle = True, drop_last= True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set) if distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data_set) if distributed else None
    return train_data_loader, valid_data_loader, train_sampler , valid_sampler , len(src_c2ix), len(trg_c2ix)


'''=======================================load dialogue data===================================================='''


def load_raw_data(sfile , tfile , password):
    data = []
    source_name = os.path.basename(sfile).replace('.zip' , '')
    target_name = os.path.basename(tfile).replace('.zip' , '')
    with zf.ZipFile(sfile, 'r') as source , zf.ZipFile(tfile , 'r') as target:
        sf = source.open(source_name, pwd=password.encode('utf-8'))
        tf = target.open(target_name, pwd=password.encode('utf-8'))
        line_counter = 0
        for sline , tline in zip(sf , tf):
            if line_counter % 10000 == 0:
                print('process line[{d%}]...'%{line_counter})
            ins = [sline.decode(), tline.decode()]
            data.append(ins)
            line_counter += 1
        sf.close()
        tf.close()
        return data


def get_sentences(lines , ndx):
    line = ' '.join(lines[ndx].split(' ')[1:])
    while ndx + 1 < len(lines) and ( not lines[ndx+1].startswith('M') and not lines[ndx+1].startswith('E')):
            line = line + ' ' + lines[ndx + 1]
            ndx += 1
    ndx += 1
    return line.strip() , ndx
if __name__ == '__main__':

    batch_size = 128
    train_precent = 0.7
    data_root = '../data/translate/small/'
    train_data_loader, valid_data_loader, input_lengths, target_lengths = get_data_loaders(\
        sfile = data_root + 'clean3.en.zip' ,
        tfile = data_root + 'clean3.zh.zip',
        batch_size = batch_size ,
        train_precent=train_precent,
        password='')
    print('end...')
