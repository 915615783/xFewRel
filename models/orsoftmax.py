# ordinary softmax model
import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from plotTool import plot2D

class OrSoftmax(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, num_classes, hidden_size=230, dropout=0.5):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(hidden_size, num_classes)
        if sentence_encoder.encoder.hidden_size < 5:
            dropout = 0
        print('dropout:', dropout)
        self.drop = nn.Dropout(dropout)

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, total_Q, label=None, is_plot=False, train_dataset=None):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        # using ordinary softmax to train
        if self.training:
            # Q = total_Q / N
            # assert int(Q) == Q  # Q must be int
            # Q = int(Q)
            Q = 25
            query_emb = self.sentence_encoder(query) # (B * Q, D)
            query = self.drop(query_emb)
            logits = self.fc1(query).view(-1, Q, self.num_classes)
            _, pred = torch.max(logits.view(-1, self.num_classes), 1)
            return logits, pred

        else:
            support_emb = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
            query_emb = self.sentence_encoder(query) # (B * total_Q, D)

            if is_plot:
                if train_dataset != None:
                    _, train_sample, train_label = train_dataset.sample_all_support()
                    train_emb = self.sentence_encoder(train_sample)
                    plot2D(support_emb, query_emb, label, N, K, total_Q, self.hidden_size, train_emb=train_emb, train_label=train_label)
                else:
                    plot2D(support_emb, query_emb, label, N, K, total_Q, self.hidden_size)

            support = self.drop(support_emb)
            query = self.drop(query_emb)
            support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
            query = query.view(-1, total_Q, self.hidden_size) # (B, total_Q, D)

            B = support.size(0) # Batch size
                
            # Prototypical Networks 
            # Ignore NA policy
            support = torch.mean(support, 2) # Calculate prototype for each class
            logits = -self.__batch_dist__(support, query) # (B, total_Q, N)
            minn, _ = logits.min(-1)
            logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
            _, pred = torch.max(logits.view(-1, N+1), 1)
            return logits, pred
