from numpy import *
from collections import Counter
import torch
import pickle
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
    seq += [0 for _ in range(max_length - len(seq))]
    return seq

def random_batch(batch_size, pairs, src_c2ix, trg_c2ix):
    input_seqs, target_seqs = [], []

    for i in random.choice(len(pairs), batch_size):
        input_seqs.append(indexes_from_text(pairs[i][0], src_c2ix))
        target_seqs.append(indexes_from_text(pairs[i][1], trg_c2ix))

    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    input_var = torch.LongTensor(input_padded).transpose(0, 1)
    # seq_len x batch_size
    target_var = torch.LongTensor(target_padded).transpose(0, 1)
    input_var = input_var.to(device)
    target_var = target_var.to(device)

    return input_var, input_lengths, target_var, target_lengths

def save_model(model , index):
    fw = open('model.' + str(index) + '.m' , 'wb')
    pickle.dump(model , fw)
    fw.flush()
    fw.close()




