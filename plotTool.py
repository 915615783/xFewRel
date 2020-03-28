import numpy as np
import matplotlib.pyplot as plt
import torch

# draw picture to find out where is the na data
# two type, support and query, we use two shape to indicate it
# Use it in model when you check the model is evaling

colors = ['blue', 'green', 'm', 'yellow', 'brown', 'red']
figcount = 0

def plot2D(support_emb, query_emb, label, N, K, total_Q, hidden_size, plot_num=2):
    '''hidden size must be 2'''
    global figcount
    support_emb = support_emb.view(-1, N, K, hidden_size).cpu().numpy() # (b, N, K, h)
    query_emb = query_emb.view(-1, total_Q, hidden_size).cpu().numpy()  # (b, total_Q, h)
    labels = label.view(-1, total_Q).cpu().numpy() # (b, total_Q)

    for i in range(plot_num):
        support = support_emb[i] # (N, K, h)
        query = query_emb[i] # (total_Q, h)
        label = labels[i] # (total_Q)
        # plot support
        for n in range(N):
            plt.scatter(support[n, :, 0], support[n, :, 1], c=colors[n], marker='^')
        # plot query
        plt.scatter(query[:, 0], query[:, 1], c=[colors[label[q]] for q in range(total_Q)], marker='.')
        plt.savefig('image/%d.png'%figcount, dpi=300)
        figcount += 1
        plt.show()


# plt.scatter([1, 2, 2], [1, 2, 1], c=['black', 'pink', 'm'], marker='^')
# plt.show()

    
