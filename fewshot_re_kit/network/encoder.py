import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim
use_simple_cnn = 1  # choose simple cnn or complicated cnn
if use_simple_cnn:
    class Encoder(nn.Module):
        def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
            nn.Module.__init__(self)
            print('using simple cnn (1 layers)')
            self.max_length = max_length
            self.hidden_size = hidden_size
            self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
            self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
            self.pool = nn.MaxPool1d(max_length)
            # self.linear = nn.Linear(10, 1)
            # self.drop = nn.Dropout(0.5)

            # For PCNN
            self.mask_embedding = nn.Embedding(4, 3)
            self.mask_embedding.weight.data.copy_(torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))
            self.mask_embedding.weight.requires_grad = False
            self._minus = -100

        def forward(self, inputs):
            return self.cnn(inputs)

        def cnn(self, inputs):
            x = self.conv(inputs.transpose(1, 2))
            x = self.pool(x)
            # x = self.drop(x)
            # x, _ = torch.topk(x, 10, dim=-1)
            # x = F.tanh(x)
            # x = self.linear(x) # n*hidden*1
            x = F.relu(x)
            return x.squeeze(2) # n x hidden_size

        def pcnn(self, inputs, mask):
            x = self.conv(inputs.transpose(1, 2)) # n x hidden x length
            mask = 1 - self.mask_embedding(mask).transpose(1, 2) # n x 3 x length
            pool1 = self.pool(F.relu(x + self._minus * mask[:, 0:1, :]))
            pool2 = self.pool(F.relu(x + self._minus * mask[:, 1:2, :]))
            pool3 = self.pool(F.relu(x + self._minus * mask[:, 2:3, :]))
            x = torch.cat([pool1, pool2, pool3], 1)
            x = x.squeeze(2) # n x (hidden_size * 3) 

else:
    # Add some cnn layers, only one layer maybe too few.
    class Encoder(nn.Module):
        def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
            nn.Module.__init__(self)
            print('using complicated cnn')

            self.max_length = max_length
            self.hidden_size = hidden_size
            self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
            self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size*6, 3, padding=1)
            self.conv2 = nn.Conv1d(self.hidden_size*6, self.hidden_size*4, 3, padding=1)
            self.conv3 = nn.Conv1d(self.hidden_size*4, self.hidden_size*2, 3, padding=1)
            self.conv4 = nn.Conv1d(self.hidden_size*2, self.hidden_size, 3, padding=1)
            self.pool = nn.MaxPool1d(max_length)

            # For PCNN
            self.mask_embedding = nn.Embedding(4, 3)
            self.mask_embedding.weight.data.copy_(torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))
            self.mask_embedding.weight.requires_grad = False
            self._minus = -100

        def forward(self, inputs):
            return self.cnn(inputs)

        def cnn(self, inputs):
            x = self.conv(inputs.transpose(1, 2))
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)
            x = self.conv4(x)
            # x = F.tanh(x)
            x = self.pool(x)
            return x.squeeze(2) # n x hidden_size

        def pcnn(self, inputs, mask):
            x = self.conv(inputs.transpose(1, 2)) # n x hidden x length
            mask = 1 - self.mask_embedding(mask).transpose(1, 2) # n x 3 x length
            pool1 = self.pool(F.relu(x + self._minus * mask[:, 0:1, :]))
            pool2 = self.pool(F.relu(x + self._minus * mask[:, 1:2, :]))
            pool3 = self.pool(F.relu(x + self._minus * mask[:, 2:3, :]))
            x = torch.cat([pool1, pool2, pool3], 1)
            x = x.squeeze(2) # n x (hidden_size * 3) 
