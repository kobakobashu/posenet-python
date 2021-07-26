# -*- coding: utf-8 -*-
"""SimpleLSTM"""

import torch
import torch as nn
import torch.nn as nn
#import torch.nn.functional as F
#import torchvision.models as models

from models.base_model import BaseModel


class Net(nn.Module):
    """Network for SimpleLSTM"""


    def __init__(self, input_size, hidden_dim, num_layers, output_size) -> None:
        """Initialization
        Args:
            input_size: input size.
            hidden_dim: Dimension of hidden state.
            num_layers: Number of RNN hidden layers.
            output_size: Output size.
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.input_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        _, (h, c) = self.lstm(x.view(len(x[0]), -1, self.input_size))
        x = self.fc(h.view(-1, self.hidden_dim))
        x = self.softmax(x)

        return x


class LSTM(BaseModel):
    """LSTM"""


    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Build model.
        Args:
            cfg: Config.
        """

        super().__init__(cfg)
        # self.vocab_size = self.cfg.model.vocab_size
        # self.emb_size = self.cfg.model.embedder.emb_size
        # self.embedder = get_embedder(self.cfg.model.embedder.name)
        #self.embedder = CBOW(vocab_size=self.cfg.model.embedder.vocab_size, emb_size=self.cfg.model.embedder.emb_size)
        #ckpt = torch.load(self.cfg.model.embedder.initial_ckpt)
        #elf.embedder.load_state_dict(ckpt['model_state_dict'])
        self.input_size = self.cfg.model.input_size
        self.hidden_dim = self.cfg.model.hidden_dim
        self.num_layers = self.cfg.model.num_layers
        self.output_size = self.cfg.model.output_size

        self.network = Net(
            # vocab_size=self.vocab_size, 
            #embedder=self.embedder, 
            input_size=self.input_size,
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers, 
            output_size=self.output_size
            )

        self.build()