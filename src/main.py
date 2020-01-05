import os
import math

from argparse import ArgumentParser

import torch
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage

from coders import *
from Seq2Seq import *
from get_loader_raw import get_data_loaders
from mylog import logger , current_time


checkpoint_dir = os.path.join('../checkpoint', current_time)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def train():
    parser = ArgumentParser()
    parser.add_argument("--source_dataset_path", type=str, default="../data/translate/small/clean3.en.1000",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--target_dataset_path", type=str, default="../data/translate/small/clean3.zh.1000",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='../cache/', help="Path or url of the dataset cache")
    parser.add_argument("--check_point", type=str, default=None, help="Path or url of the dataset cache")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Batch size for validation")
    parser.add_argument("--hidden_dim", type=int, default=100, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--train_precent", type=float, default=0.7, help="Batch size for validation")
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--log_step", type=int, default=10, help="Multiple-choice loss coefficient")
    parser.add_argument("--raw_data", action='store_true', default=True, help="If true read data by raw function")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()
    logger.info(args)

    device = torch.device(args.device)

    #跳过编号为0的卡
    if args.local_rank != -1:
       args.local_rank += 1
    args.distributed = (args.local_rank != -1)

    #分布式的初始化要在get_data_loaders前，get_data_loaders中DistributedSampler使用了分布试信息
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')


    train_data_loader, valid_data_loader, train_sampler , valid_sampler , input_lengths, target_lengths = \
        get_data_loaders(args.source_dataset_path, \
                         args.target_dataset_path, \
                         args.batch_size, \
                         args.train_precent ,\
                         args.raw_data,\
                         args.distributed)

    encoder = Encoder(input_lengths + 1, args.embedding_dim, args.hidden_dim)
    decoder = Decoder(target_lengths + 1, args.embedding_dim, args.hidden_dim)
    model = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss(ignore_index=0).to(device)

    # Initialize distributed training if needed


    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


    if args.check_point is not None:
        logger.info('load checkpoint from %s'%args.check_point)
        check_point = torch.load(args.check_point)
        model.load_state_dict(check_point)

    def update(engine, batch):
        model.train()
        src_seqs = batch[0].transpose(0, 1).to(device)
        src_lengths = batch[1].to(device)
        trg_seqs = batch[2].transpose(0, 1).to(device)
        output = model(src_seqs, src_lengths, trg_seqs)
        loss = criterion(output.contiguous().view(-1, output.shape[2]), trg_seqs.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    trainer = Engine(update)

    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            src_seqs = batch[0].transpose(0, 1).to(device)
            src_lengths = batch[1].to(device)
            trg_seqs = batch[2].transpose(0, 1).to(device)
            output = model(src_seqs, src_lengths, trg_seqs)
            return output.contiguous().view(-1, output.shape[2]), trg_seqs.contiguous().view(-1)

    evaluator = Engine(inference)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(criterion, output_transform=lambda x: (x[0], x[1])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0], x[1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)



    steps = len(train_data_loader.dataset) // train_data_loader.batch_size
    steps = steps if steps > 0 else 1
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=steps)

    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))



    '''======================early stopping =========================='''
    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        return -val_loss
    handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    '''==================print information by iterator========================='''
    steps = len(train_data_loader.dataset) // train_data_loader.batch_size
    steps = steps if steps > 0 else 1
    logger.info('steps:%d' % steps)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_data_loader)
        ms = evaluator.state.metrics
        logger.info("Validation Results - Epoch: [{}/{}]  Avg accuracy: {:.6f} Avg loss: {:.6f}"
                    .format(trainer.state.epoch, trainer.state.max_epochs, ms['accuracy'], ms['nll']))

    #单卡local_rank等于1，多卡时，只在第一个卡上打印信息。
    # -1:单卡时的情况，1：多卡时第一个卡（注意：0号卡在前面跳过去了，所以这时是1，不是0）
    if args.local_rank in [-1, 1]:

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            if trainer.state.iteration % args.log_step == 0:
                logger.info("Epoch[{}/{}] Step[{}/{}] Loss: {:.6f}".format(trainer.state.epoch,
                                                                           trainer.state.max_epochs,
                                                                           trainer.state.iteration % steps,
                                                                           steps,
                                                                           trainer.state.output * args.gradient_accumulation_steps)
                            )

        '''================add check point========================'''
        logs_dir = checkpoint_dir + '/' + str(args.local_rank)
        checkpoint_handler = ModelCheckpoint(logs_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

    '''==============run trainer============================='''
    trainer.run(train_data_loader, max_epochs=args.n_epochs)


if __name__ == '__main__':
    train()
