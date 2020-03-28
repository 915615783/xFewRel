import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from plotTool import plot2D
class Siamese(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230, dropout=0):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.normalize = nn.LayerNorm(normalized_shape=hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, support, query, N, K, total_Q, label=None, is_plot=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        support = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query) # (B * total_Q, D)

        if is_plot:
            plot2D(support, query, label, N, K, total_Q, self.hidden_size)

        # Layer Norm
        support = self.normalize(support)
        query = self.normalize(query)
        
        # Dropout ?
        support = self.drop(support)
        query = self.drop(query)

        support = support.view(-1, N * K, self.hidden_size) # (B, N * K, D)
        query = query.view(-1, total_Q, self.hidden_size) # (B, total_Q, D)
        B = support.size(0) # Batch size
        support = support.unsqueeze(1) # (B, 1, N * K, D)    liu add: 这个是element-wise乘法，简单粗暴呀，我以为要算距离的。
        query = query.unsqueeze(2) # (B, total_Q, 1, D)      liu add: 不对，这里求得是余弦距离，内积。上面规范化了之后就不需要除东西，直接内积就是余弦距离

        #  Dot production
        z = (support * query).sum(-1) # (B, total_Q, N * K)
        z = z.view(-1, total_Q, N, K) # (B, total_Q, N, K)

        # Max combination
        logits = z.max(-1)[0] # (B, total_Q, N)

        # NA
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)

        _, pred = torch.max(logits.view(-1, N+1), 1)
        return logits, pred 
