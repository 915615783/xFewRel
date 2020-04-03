import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns

# draw picture to find out where is the na data
# two type, support and query, we use two shape to indicate it
# Use it in model when you check the model is evaling

colors = ['blue', 'green', 'm', 'yellow', 'brown', 'red']
figcount = 0

def plot2D(support_emb, query_emb, label, N, K, total_Q, hidden_size, plot_num=2, train_emb=None, train_label=None):
    '''hidden size must be 2'''
    global figcount
    is_plot_train = False
    support_emb = support_emb.view(-1, N, K, hidden_size).cpu().numpy() # (b, N, K, h)
    query_emb = query_emb.view(-1, total_Q, hidden_size).cpu().numpy()  # (b, total_Q, h)
    labels = label.view(-1, total_Q).cpu().numpy() # (b, total_Q)
    if isinstance(train_emb, torch.Tensor):
        train_emb = train_emb.view(64, -1, hidden_size).cpu().numpy()  # (64, n, h)
        train_label = train_label.view(64, -1).cpu().numpy()    # (64, n)
        is_plot_train = True
        

    if not os.path.exists('image/'):
        os.makedirs('image/')

    for i in range(plot_num):
        if is_plot_train:
            g = sns.jointplot(train_emb[:, :, 0].reshape(-1), train_emb[:, :, 1].reshape(-1), kind='kde')  # or kind=kde,scatter
            ax = plt.gcf().axes[0]
        else:
            ax = plt.gca()
        support = support_emb[i] # (N, K, h)
        query = query_emb[i] # (total_Q, h)
        label = labels[i] # (total_Q)
        # plot support
        for n in range(N):
            ax.scatter(support[n, :, 0], support[n, :, 1], c=colors[n], marker='^')
        # plot query
        ax.scatter(query[:, 0], query[:, 1], c=[colors[label[q]] for q in range(total_Q)], marker='.')
        plt.savefig('image/%06d.png'%figcount, dpi=150)
        figcount += 1
        plt.clf()
        plt.close('all')


# plt.scatter([1, 2, 2], [1, 2, 1], c=['black', 'pink', 'm'], marker='^')
# plt.show()

    
