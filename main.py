import argparse
import torch
import shutil
import numpy as np
import random
from torch.autograd import Variable
from torch.optim import lr_scheduler
from models.CNN import CNN
from models.BiLSTM import BiLSTM
from models.CNN_BiLSTM import CNN_BiLSTM
from models.CNN_BiLSTM_ATT import CNN_BiLSTM_ATTENTION
from DataLoader import DataLoader
from utils import *

import sys

#use_cuda = False
def parse_args():
    parser = argparse.ArgumentParser(description="Neural model for Translation Quality Classification")
    parser.add_argument('-M','--model', type = str, default = "CNN")
    parser.add_argument('-Opt','--optimizer', type = str, default = "adam")
    parser.add_argument('-B','--batch_size', type = int, default= 64)
    parser.add_argument('-V', '--vocab_size', type = int, default= 30000)
    parser.add_argument('-D','--embed_dim', type = int, default = 300)
    parser.add_argument('-Ci','--in_channels', type = int, default= 1)
    parser.add_argument('-Ks','--kernel_sizes', type=list, default=[3,4,5])
    parser.add_argument('-Kn','--kernel_nums', type = int, default= 200 )
    parser.add_argument('-Hd','--hidden_dim', type = int, default= 100 )
    parser.add_argument('-LYR','--num_layers', type = int, default= 2 )
    parser.add_argument('-Bi','--bidirectional', type = bool, default= True)
    parser.add_argument('-MSL','--max_seq_len', type = int, default= 60)
    parser.add_argument('-LR','--learning_rate', type = float, default= 0.8)
    parser.add_argument('-E','--num_epochs', type = int, default= 100)
    parser.add_argument('-DR','--dropout', type=float, default=0.3)
    parser.add_argument('-C','--num_class', type=int, default=2)
    parser.add_argument('-TRN','--train_file', default= './data/cwmt_train_comb.txt')
    parser.add_argument('-DEV','--dev_file', default= './data/cwmt_dev_comb.txt')
    parser.add_argument('-TST','--test_file', default= './data/htqe_test_comb.txt')
    parser.add_argument('-Emb','--pretrained_embeddings', type=str, default= './word2vec/wiki.en_zh.vec')
    parser.add_argument('-R','--prediction_file', type=str, default= './predictions/htqe_test_comb.txt')
    parser.add_argument('-S','--saved_model', type=str, default= './saved_model/cwmt.en_zh.pt')
    parser.add_argument('-VAL','--test', default=False, type=lambda x: (str(x).lower() == 'true'), help='whether to train to load pretrained models')
    parser.add_argument('-L2','--weight_decay', type=float, default=1e-3, help='setting up a float')
    parser.add_argument('--grad_clip', type=float, default=3e-1,help='grad_clip')
    parser.add_argument('--seed_num', type=int, default=123)
    parser.add_argument('--momentum', type=float, default=0.3)
    parser.add_argument('--scheduler', type=str, help='steplr, lambdalr, multisteplr, reducelronplateau')
    parser.add_argument('--use_gpu', action='store_false',help='use gpu device')
    #parser.add_argument('--disable_cuda', action='store_false', help='switch between CPU and GPU device')
    parser.add_argument('--run_log',type=str, default= 'quality_estimation', required=True)

    args = parser.parse_args()


    return args

 

def main():
    args=parse_args()
    train_iter = DataIterator(args.train_file, args.max_seq_len, args.batch_size)
    dev_iter = DataIterator(args.dev_file, args.max_seq_len, args.batch_size)
    test_iter = DataIterator(args.test_file, args.max_seq_len, args.batch_size)

    train_data = DataLoader(args.train_file)
    vocab_size = len(train_data.vocab)
    weights = train_data.get_pretrained_weights(args.pretrained_embeddings)
    args.pretrained_weights = weights
    # udpate the args
    args.vocab_size = vocab_size
    # a small step to solve the problem of command line input interpreted as string
    args.kernel_sizes =[int(x) for x in args.kernel_sizes]


    logger= getLogger(args.run_log)

    print("\nParameters:")
    
    for attr, value in sorted(args.__dict__.items()):
        logger.info("\t{}={}".format(attr.upper(), value))

    if args.model=="CNN":
        model = CNN(args)
        #model = CNN(args.vocab_size, args.embed_dim,args.max_seq_len, args.num_class, args.kernel_nums, args.dropout, args.kernel_sizes, args.pretrained_weights, args.use_gpu)
    elif args.model =="BiLSTM":
        model = BiLSTM(args)
        #model = BiLSTM(args.vocab_size, args.embed_dim, args.hidden_dim,args.num_layers,args.bidirectional, args.pretrained_weights,args.num_class, args.use_gpu)
    elif args.model =="CNN_BiLSTM":
        model =CNN_BiLSTM(args)
        #model = CNN_BiLSTM(args.hidden_dim, args.num_class, args.num_layers, args.vocab_size, args.embed_dim, args.kernel_nums, args.kernel_sizes,args.dropout,args.pretrained_weights, args.use_gpu )
    elif args.model =="CNN_BiLSTM_ATT":
        model = CNN_BiLSTM_ATTENTION(args)
    print(model)

    # starting training, comment this line if you are load a pretrained model
    if args.test is False:
        model.train()
        ##
        train (train_iter, dev_iter, test_iter, model, args, logger)
    
    else:
        model = torch.load(args.saved_model, map_location=lambda storage, loc: storage)
        model.eval()
        eval(test_iter, model, args, logger)





if __name__ == "__main__":
    main()