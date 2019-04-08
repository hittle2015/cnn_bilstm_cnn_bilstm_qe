import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    """
     vocab_size, embed_size, num_hiddens, num_layers
    """
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.args = args
        self.num_hiddens = args.hidden_dim
        self.num_layers = args.num_layers
        self.use_gpu = args.use_gpu
        self.bidirectional = args.bidirectional
        self.embedding = nn.Embedding.from_pretrained(args.pretrained_weights)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=args.embed_dim, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=args.dropout)
        if self.bidirectional:
            self.decoder = nn.Linear(self.num_hiddens * 4, args.num_class)
        else:
            self.decoder = nn.Linear(self.num_hiddens * 2, args.num_class)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs