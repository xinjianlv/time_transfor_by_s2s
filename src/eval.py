from argparse import ArgumentParser
from torch import optim
from torch import nn
import torch
import pdb
import json
import pickle
import numpy
import socket
from datetime import datetime
import os


from utils import *
from coders import *
from Seq2Seq import *
from  mylog import logger
from get_loader_raw import get_data_loaders , build_vocab_raw,load_raw_data
def test():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../data/xiaohuangji/xiaohuangji50w_nofenci.seg.conv",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--check_point", type=str, default='../checkpoint/Nov05_20-45-33/checkpoint_mymodel_8.pth', help="Path or url of the dataset cache")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for validation")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Batch size for validation")
    parser.add_argument("--hidden_dim", type=int, default=100, help="Batch size for validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--train_precent", type=float, default=0.7, help="Batch size for validation")
    args = parser.parse_args()
    device = torch.device(args.device)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join('../logs', current_time + '_' + socket.gethostname())


    train_data_loader, valid_data_loader, input_lengths, target_lengths = get_data_loaders(args.dataset_path, args.batch_size, args.train_precent,True)

    encoder = Encoder(input_lengths + 1, args.embedding_dim, args.hidden_dim)
    decoder = Decoder(target_lengths + 1, args.embedding_dim, args.hidden_dim)
    model = Seq2Seq(encoder, decoder, device).to(device)

    check_point = torch.load(args.check_point)
    model.load_state_dict(check_point)
    model.eval()

    pairs = load_raw_data(args.dataset_path)
    data = array(pairs)
    src_texts = data[:, 0]
    trg_texts = data[:, 1]
    src_c2ix, src_ix2c = build_vocab_raw(src_texts)
    trg_c2ix, trg_ix2c = build_vocab_raw(trg_texts)

    def get_decode(src):
        result = []
        for t in src:
            result.append(src_ix2c[t])
        sndx = 0
        if '<s>' in result:
            sndx = result.index('<s>') + 1
        endx = result.index('<e>')
        return ''.join(result[sndx:endx])

    def get_decode_target(target):
        result = []
        for t in target:
            result.append(trg_ix2c[int(t)])
        sndx = 0
        if '<s>' == result[0]:
            sndx = result.index('<s>') + 1
        endx = len(result) - 1
        if '<e>' in result:
            endx = result.index('<e>')
        return ''.join(result[sndx:endx])

    max_src_len = max(list(map(len, src_texts))) + 2
    max_trg_len = max(list(map(len, trg_texts))) + 2
    max_src_len, max_trg_len

    for batch in valid_data_loader:
        src_seqs = batch[0].transpose(0, 1).to(device)
        src_lengths = batch[1].to(device)
        trg_seqs = batch[2].transpose(0, 1).to(device)
        outputs, attn_weights = model.predict(src_seqs = src_seqs ,src_lengths = src_lengths)
        # print(outputs.cpu().detach().numpy())
        outputs_index = torch.argmax(outputs.cpu(), dim=2)
        outputs_index_mat = outputs_index.permute(1,0)

        for i in range(outputs_index_mat.shape[0]):
            logger.info('src:    \t%s'%get_decode(src_seqs.cpu().permute(1,0)[i].numpy()))
            logger.info('target :\t%s'%get_decode_target(trg_seqs.cpu().permute(1,0)[i].numpy()))
            logger.info('predict:\t%s'%get_decode_target(outputs_index_mat[i].detach().numpy()[1:]))
            logger.info('='*64)

if __name__ == '__main__':
    test()