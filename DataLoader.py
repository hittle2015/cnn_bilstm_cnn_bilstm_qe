#! /usr/bin/env python
import codecs
import os
import torch
import time

import numpy as np
from itertools import chain
import torch.utils.data.dataloader as dataloader


class DataLoader(object):
    def __init__(self, file_path,pretrained_emb=None, *kwargs):
        self.src_data = []
        self.tgt_data = []
        self.text = []
        self.scores = []
        self.vocab = ['unk']
        self.word2idx = {}
        self.idx2word = {}
        self.pretrained_emb = pretrained_emb
        with codecs.open(file_path,'r','utf-8') as f:
            for i, line in enumerate(f):
                try:
                    info  = line.strip().split('\t')
                    assert len(info) == 4, line
                    src = info[1].split()
                    tgt = info[2].split()
                    src_tgt = src + tgt
                    score = int(info[3])
                    self.src_data.append(src)
                    self.tgt_data.append(tgt)
                    self.text.append(src_tgt)
                    self.scores.append(score)
                except ValueError as e:
                    print ("error",e,"on line",i, info[0])
        src_vocab= set(chain(*self.src_data))
        tgt_vocab = set(chain(*self.tgt_data))
        ## combine the src and tgt vocabulary for the convenience of joint training with a combined pretrained embedding
        self.vocab.extend(set(chain(src_vocab, tgt_vocab)))
        self.word2idx.update({word: i for i, word in enumerate(self.vocab)})
        self.idx2word.update({i: word for i, word in enumerate(self.vocab)})
    def _encoding(self,tokenized_samples):
        """
        input: tokenized texts of src and tgt sentences
        output: indexes of  the src and tgt in the vocabulary dictionary
        e.g. input "I have a bag . 我 有 一个 包 。" shall return
                [[41424, 31383, 21800, 6858, 11586, 39246, 40689, 42498, 27937, 17831]]
             
        """
        features = []
        for sample in tokenized_samples:
            feature = []
            for token in sample:
                if token in self.word2idx:
                    feature.append(self.word2idx[token])
                else:
                    feature.append(0)
            features.append(feature)
        return features

    def _padding(self,features, maxlen=50, PAD=0, *kwargs):
        """
        input: src sequences, tgt sequences 
        output: padded sequence equal to the maximum length
        I decided to set a max length for making all inputs equal length instead of align them with the max length of the sequences which may be ridiculously long,\
        as long as 209 words.
        """
        #         max_src_seq_len = max(len(s) for s in self.src_data)
        #         max_tgt_seq_len = max(len(t) for t in self.tgt_data)
        #         if max_src_seq_len >= max_tgt_seq_len and max_src_seq_len >= maxlen:
        #             maxlen = max_src_seq_len
        #         elif max_src_seq_len >= max_tgt_seq_len and max_src_seq_len< maxlen:
        #             maxlen = maxlen
        #         elif max_tgt_seq_len >= max_src_seq_len and max_tgt_seq_len >= maxlen:
        #             maxlen = max_tgt_seq_len
        #         elif max_tgt_seq_len >= max_src_seq_len and max_tgt_seq_len < maxlen:
        #             maxlen = maxlen 
        padded_features = []
        for feature in features:
            if len(feature) >= maxlen:
                padded_feature = feature[:maxlen]
            else:
                padded_feature = feature
                while(len(padded_feature) < maxlen):
                    padded_feature.append(PAD)
            padded_features.append(padded_feature)
        return padded_features


    def get_pretrained_weights(self,pretrained_emb_path, emb_dim=300, *kwargs):
        """
        gensim module to load the pretrained vectors seems intolerably slow. Therefore we use a customized version
        """
        #         wvmodel= gensim.models.KeyedVectors.load_word2vec_format(pretrained_emb_path, binary=False)
        #         weights = torch.zeros(len(self.vocab), emb_dim)
        #         for i in range(len(wvmodel.index2word)):
        #             try:
        #                 index = self.word2idx[wvmodel.index2word[i]]
        #             except:
        #                 continue
        #             weight[index, :] = torch.from_numpy(wvmodel.get_vector(
        #                 self.idx2word[word_to_idx[wvmodel.index2word[i]]]))
        #         return weights
        weights = torch.zeros(len(self.vocab), emb_dim)
        with codecs.open(pretrained_emb_path,'r','utf-8') as f:
            next(f)
            for line in f:
                info = line.strip().split()
                if len(info)!= (emb_dim +1):
                    continue
                try:
                    word, data = info[0], info[1:]
                except IndexError as e:
                    continue
                if word in self.word2idx:
                    weights[self.word2idx[word]] = torch.FloatTensor(np.asarray(data, dtype=np.float32))
        return weights
        
