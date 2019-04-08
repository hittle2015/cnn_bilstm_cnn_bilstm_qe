import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import numpy as np
# import random

# torch.manual_seed(args.seed_num)
# random.seed(seed_num)


"""
    Neural Network: CNN_BiLSTM
    Detail: cnn model and LSTM model train on the input independly, then the result of both concat
"""

class CNN_BiLSTM(nn.Module):

    """
     Args: hidden_dim, num_layers, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, use_gpu,pretrained_weights
    """
    def __init__(self, args):
        super(CNN_BiLSTM, self).__init__()
        self.args = args
        V = args.vocab_size
        D = args.embed_dim
        C = args.num_class
        self.C = C
        Ci = 1
        Co = args.kernel_nums
        Ks = args.kernel_sizes
        self.bidirectional=args.bidirectional
        self.embed = nn.Embedding(V, D)
        # pretrained  embedding
        if args.pretrained_embeddings is not None:
            self.embed.weight.data.copy_(args.pretrained_weights)
        self.embed.weight.requires_grad = False

        # CNN
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), padding=(K//2, 0), stride=1) for K in Ks])
        #print(self.convs1)
        self.dropout = nn.Dropout(args.dropout)


        #BiLSTM

        self.encoder = nn.LSTM(input_size=D, hidden_size=args.hidden_dim,
                               num_layers=args.num_layers, bidirectional=args.bidirectional,
                               dropout=0)

        self.linear_layers = []
        for _ in range(args.num_layers - 1):
            self.linear_layers.append(nn.Linear(args.hidden_dim*4, args.hidden_dim))
        self.linear_layers = nn.ModuleList(self.linear_layers)

        if self.bidirectional:
            self.dimcast = nn.Linear((len(Ks)*Co + args.hidden_dim), 4*args.hidden_dim)
            self.decoder = nn.Linear(args.hidden_dim* 4, C)
        else:
            self.dimcast = nn.Linear((len(ks)*Co + args.hidden_dim), 2*args.hidden_dim)
            self.decoder = nn.Linear(args.hidden_dim * 2, C)

        #
    def forward(self,x):
        cnn_x = self.embed(x)  # (N, W, D)
        #static
        cnn_x = Variable(cnn_x)
        cnn_x = cnn_x.unsqueeze(1)# (N, Ci, W, D)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1] # [(N, Co, W), ...]*len(Ks)
        cnn_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_x]  # [(N, Co), ...]*len(Ks)

        cnn_x  = torch.cat(cnn_x, 1)
        cnn_out = self.dropout(cnn_x) # (N, len(Ks)*Co)
        #print('5', cnn_out.shape)

        bilstm_x = self.embed(x)
        #print('1', bilstm_x.shape)
        states, hidden = self.encoder(bilstm_x.permute([1, 0, 2]))
        
        bilstm_x = torch.cat([states[0], states[-1]], dim=1)
        #print('2', bilstm_x.shape)
        for layer in self.linear_layers:
            bilstm_out = layer(bilstm_x)
            
        #print('3', bilstm_out.shape)

        cnn_bilstm_out = torch.cat((cnn_out, bilstm_out), dim=1)
        #print('4', cnn_bilstm_out.shape)
        cnn_bilstm_feature = self.dimcast(cnn_bilstm_out)
        logit = self.decoder(cnn_bilstm_feature)

        return logit






