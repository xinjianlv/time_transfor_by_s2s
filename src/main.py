import os

from argparse import ArgumentParser

import torch
from torch import nn
from torch import optim

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage

from coders import *
from Seq2Seq import *
from get_loader import get_data_loaders
from mylog import logger , current_time


checkpoint_dir = os.path.join('../checkpoint', current_time)


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../data/time_transfor/Time Dataset.json",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='../cache/', help="Path or url of the dataset cache")
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
    args = parser.parse_args()
    device = torch.device(args.device)




    train_data_loader, valid_data_loader, input_lengths, target_lengths = get_data_loaders(args.dataset_path, args.batch_size, args.train_precent)

    encoder = Encoder(input_lengths + 1, args.embedding_dim, args.hidden_dim)
    decoder = Decoder(target_lengths + 1, args.embedding_dim, args.hidden_dim)
    model = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss(ignore_index=0).to(device)

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
    metrics = {"nll": Loss(criterion, output_transform=lambda x: (x[0], x[1])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0], x[1]))}
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    Loss(criterion, output_transform=lambda x: (x[0], x[1]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_data_loader)
        ms = evaluator.state.metrics
        logger.info("Validation Results - Epoch: [{}/{}]  Avg accuracy: {:.6f} Avg loss: {:.6f}"
              .format(trainer.state.epoch,trainer.state.max_epochs, ms['accuracy'], ms['nll']))

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
    checkpoint_handler = ModelCheckpoint(checkpoint_dir, 'checkpoint', save_interval=1, n_saved=3)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

    '''==============run trainer============================='''
    trainer.run(train_data_loader, max_epochs=args.n_epochs)


if __name__ == '__main__':
    train()
