from transformers import BertTokenizer, BertModel

# m = BertModel.from_pretrained('pretrain\\bert-base-uncased')

# print(m)
t = BertTokenizer.from_pretrained('pretrain\\bert-base-uncased\\vocab.txt')
print(t)






























# # define a dataset class
# # the point is to finish a new __getitem__() that can support ordinary softmax training
# # return one type of instance instead of both support_set and query_set
# # then the label include all the classes of the labels.

# # the ultimate purpose is to find out if siamese is necessary.

# import torch
# import torch.utils.data as data
# import os
# import numpy as np
# import random
# import json

# class FewRelDatasetForNormalSoftmax(data.Dataset):
#     def __init__(self, name, encoder, N, K, Q, na_rate, root='./data'):
#         '''
#         name indicate the file name without .json
#         In this case, we don't consider the NOTA situation.
#         '''
#         self.root = root
#         path = os.path.join(root, name + ".json")
#         if not os.path.exists(path):
#             print("[ERROR] Data file does not exist!")
#             assert(0)
#         self.json_data = json.load(open(path))
#         self.classes = list(self.json_data.keys())
#         self.class2id = {c:i for i, c in enumerate(self.classes)}
#         self.N = N
#         self.K = K
#         self.Q = Q
#         self.na_rate = na_rate
#         self.encoder = encoder
#         self.json_data_with_relation = []
#         for r in self.json_data.keys():
#             for instance in self.json_data[r]:
#                 instance['rel'] = r
#                 self.json_data_with_relation.append(instance)

#     def __getraw__(self, item):
#         word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
#             item['h'][2][0],
#             item['t'][2][0])
#         return word, pos1, pos2, mask 

#     def __additem__(self, d, word, pos1, pos2, mask):
#         d['word'].append(word)
#         d['pos1'].append(pos1)
#         d['pos2'].append(pos2)
#         d['mask'].append(mask)

#     def __getitem__(self, index):
#         support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
#         query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
#         query_label = []
        
#         queries = np.random.choice(self.json_data_with_relation, self.Q)
#         for q in queries:
#             word, pos1, pos2, mask = self.__getraw__(q)
#             word = torch.tensor(word).long()
#             pos1 = torch.tensor(pos1).long()
#             pos2 = torch.tensor(pos2).long()
#             mask = torch.tensor(mask).long()
#             label = self.class2id[q['rel']]
#             self.__additem__(query_set, word, pos1, pos2, mask)
#             query_label.append(label)
#         return support_set, query_set, query_label



# # d = FewRelDatasetForNormalSoftmax('train_wiki', None, 1, 1, 1, 0)
# # print(d.classes)
# # print(d.class2id)
# # print(len(d.json_data))

# # v = FewRelDatasetForNormalSoftmax('val_wiki', None, 1, 1, 1, 0)
# # print(v.classes)
# # print(len(v.json_data))
# # for i in v.classes:
# #     print(i, i in d.classes)


# #moudle.train can check if the model is eval or train

