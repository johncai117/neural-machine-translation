#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    python main.py --mode train --train_src <train_src_file> --train_tgt <train_tgt_file> \
        --dev_src <dev_src_file> --dev_tgt <dev_tgt_file> --vocab_file <vocab_file>
    python main.py --mode decode --test_src <test_src_file> --model_path <model_path> --output_file <output_file>
    python main.py --mode decode --test_src <test_src_file> --test_tgt <test_tgt_file> \
        --model_path <model_path> --output_file <output_file>

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train_src <file>                      train source file
    --train_tgt <file>                      train target file
    --dev_src <file>                        dev source file
    --dev_tgt <file>                        dev target file
    --vocab_file <file>                          vocab file
    --seed <int>                            seed [default: 123]
    --batch_size <int>                      batch size [default: 32]
    --embed_size <int>                      embedding size [default: 64]
    --hidden_size <int>                     hidden size [default: 64]
    --clip_grad <float>                     gradient clipping [default: 5.0]
    --log_every <int>                       log every [default: 10]
    --max_epoch <int>                       max epoch [default: 50]
    --patience <int>                        wait for how many iterations to decay learning rate [default: 5]
    --max_num_trial <int>                   terminate training after how many trials [default: 5]
    --lr_decay <float>                      learning rate decay [default: 0.5]
    --beam_size <int>                       beam size [default: 5]
    --lr <float>                            learning rate [default: 0.001]
    --uniform_init <float>                  uniformly initialize all parameters [default: 0.1]
    --save_to <file>                        model save path [default: model.bin]
    --valid_niter <int>                     perform validation after how many iterations [default: 100]
    --dropout <float>                       dropout [default: 0.3]
    --max_decoding_time_step <int>          maximum number of decoding time steps [default: 70]
"""

import math
import time
import argparse

from nmt_model import Hypothesis, NMT
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from typing import List
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab
import pandas as pd

import torch
import torch.nn.utils


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def train(args):
    """ Train the NMT Model.
    """
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')
    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    vocab = Vocab.load(args.vocab_file)
    model = NMT(embed_size=args.embed_size,
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout,
                vocab=vocab)
    model.train()

    if np.abs(args.uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init))
        for p in model.parameters():
            p.data.uniform_(-args.uniform_init, args.uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('use device: %s' % device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    iter_list = []
    iter_list2 = []
    train_ppl_list = []
    val_ppl_list = []
    while True:
        epoch += 1
        batch_num = math.ceil(len(train_data) / args.batch_size)
        current_iter = 0
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=args.batch_size, shuffle=True):
            current_iter += 1
            train_iter += 1

            optimizer.zero_grad()
            batch_size = len(src_sents)
            example_losses = -model(src_sents, tgt_sents)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size
            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            # omitting leading `<s>`
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % args.log_every == 0:
                print('epoch %d (%d / %d), iter %d, avg. loss %.2f, avg. ppl %.2f '
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' %
                      (epoch, current_iter, batch_num, train_iter,
                       report_loss / report_examples,
                       math.exp(report_loss / report_tgt_words),
                       cum_examples,
                       report_tgt_words / (time.time() - train_time),
                       time.time() - begin_time))


                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % args.valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                      cum_loss / cum_examples,
                      np.exp(cum_loss / cum_tgt_words),
                      cum_examples))
                iter_list.append(train_iter)
                train_ppl_list.append(np.exp(cum_loss / cum_tgt_words))

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...')

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl))
                iter_list2.append(train_iter)
                val_ppl_list.append(dev_ppl)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('epoch %d, iter %d: save currently the best model to [%s]' %
                          (epoch, train_iter, args.model_path))
                    model.save(args.model_path)
                    torch.save(optimizer.state_dict(), args.model_path + '.optim')
                elif patience < args.patience:
                    patience += 1
                    print('hit patience %d' % patience)

                    if patience == args.patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial)
                        if num_trial == args.max_num_trial:
                            print('early stop!')
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                        print('load previously best model and decay learning rate to %f' % lr)

                        # load model
                        params = torch.load(args.model_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers')
                        optimizer.load_state_dict(torch.load(args.model_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

            if epoch == args.max_epoch:
                print('reached maximum number of epochs!')
                ##df = pd.DataFrame({'iter_list1': iter_list, 'iter_list2': iter_list2,  'train': train_ppl_list, 'val': val_ppl_list})
                ##df.to_csv('NEW_train_val_ppl.csv')
                exit(0)

def decode(args):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    print("load test source sentences from [{}]".format(args.test_src))
    test_data_src = read_corpus(args.test_src, source='src')
    if args.test_tgt:
        print("load test target sentences from [{}]".format(args.test_tgt))
        test_data_tgt = read_corpus(args.test_tgt, source='tgt')

    print("load model from {}".format(args.model_path))
    model = NMT.load(args.model_path)

    if args.cuda:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=args.beam_size,
                             max_decoding_time_step=args.max_decoding_time_step)

    if args.test_tgt:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100))

    with open(args.output_file, 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def beam_search(model: NMT, test_data_src: List[List[str]],
                beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding'):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size,
                                             max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)

    if was_training:
        model.train(was_training)

    return hypotheses


if __name__ == '__main__':
    # Check pytorch version
    assert(torch.__version__ == "1.3.0"), \
        "Please update your installation of PyTorch. " \
        "You have {} and you should have version 1.3.0".format(torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, choices=["train", "decode"])
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--train_src", default=None, type=str, help="train source file")
    parser.add_argument("--train_tgt", default=None, type=str, help="train target file")
    parser.add_argument("--dev_src", default=None, type=str, help="dev source file")
    parser.add_argument("--dev_tgt", default=None, type=str, help="dev target file")
    parser.add_argument("--test_src", default=None, type=str, help="test source file")
    parser.add_argument("--test_tgt", default=None, type=str, help="test target file")
    parser.add_argument("--vocab_file", default=None, type=str)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--embed_size", default=64, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--clip_grad", default=5.0, type=float)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--patience", default=5, type=int,
                        help="wait for how many iterations to decay learning rate")
    parser.add_argument("--max_num_trial", default=5, type=int,
                        help="terminate training after how many trials")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--lr_decay", default=0.5, type=float, help="learning rate decay")
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--uniform_init", default=0.1, type=float, help="uniformly initialize all parameters")
    parser.add_argument("--model_path", default="model.bin", type=str)
    parser.add_argument("--valid_niter", default=100, type=int,
                        help="perform validation after how many iterations")
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--max_decoding_time_step", default=70, type=int,
                        help="maximum number of decoding time steps")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed * 13 // 7)

    if args.mode == "train":
        assert args.train_src
        assert args.train_tgt
        assert args.dev_src
        assert args.dev_tgt
        assert args.vocab_file
        assert args.model_path
        train(args)
    elif args.mode == "decode":
        assert args.test_src
        assert args.model_path
        assert args.output_file
        decode(args)
    else:
        raise RuntimeError("invalid mode: {}".format(args.mode))
