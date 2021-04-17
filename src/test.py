import os
import io
import numpy as np
import sympy as sp
import torch

from src.utils import AttrDict
from src.envs import build_env
from src.model import build_modules

from src.envs.sympy_utils import simplify

from torch.utils.data import DataLoader
from functools import partial


def read_data(path):
    with io.open(path, mode='r', encoding='utf-8') as f:
        lines = [line.rstrip().split('|') for line in f]
        data = [xy.split('\t') for _, xy in lines]
        data = [xy for xy in data if len(xy) == 2]
    return data


def batch_sequences(sequences, env):
    """
    Take as input a list of n sequences (torch.LongTensor vectors) and return
    a tensor of size (slen, n) where slen is the length of the longest
    sentence, and a vector lengths containing the length of each sentence.
    """
    lengths = torch.LongTensor([len(s) + 2 for s in sequences])
    sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(env.pad_index)
    assert lengths.min().item() > 2

    sent[0] = env.eos_index
    for i, s in enumerate(sequences):
        sent[1:lengths[i] - 1, i].copy_(s)
        sent[lengths[i] - 1, i] = env.eos_index

    return sent, lengths


def collate_fn(elements):
    """
    Collate samples into a batch.
    """
    x, y = zip(*elements)
    nb_ops = [sum(int(word in env.OPERATORS) for word in seq) for seq in x]
    x = [torch.LongTensor([env.word2id[w] for w in seq if w in env.word2id]) for seq in x]
    y = [torch.LongTensor([env.word2id[w] for w in seq if w in env.word2id]) for seq in y]
    x, x_len = batch_sequences(x, env)
    y, y_len = batch_sequences(y, env)
    return (x, x_len), (y, y_len), torch.LongTensor(nb_ops)


path = './sample_data/prim_fwd.txt'
data = read_data(path)
for i in range(len(data)):
    data[i] = tuple([sent.split(" ") for sent in data[i]])

params = params = AttrDict({

    # environment parameters
    'env_name': 'char_sp',
    'int_base': 10,
    'balanced': False,
    'positive': True,
    'precision': 10,
    'n_variables': 1,
    'n_coefficients': 0,
    'leaf_probs': '0.75,0,0.25,0',
    'max_len': 512,
    'max_int': 5,
    'max_ops': 15,
    'max_ops_G': 15,
    'clean_prefix_expr': True,
    'rewrite_functions': '',
    'tasks': 'prim_fwd',
    'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1',
})
env = build_env(params)
loader = DataLoader(data, batch_size=32, shuffle=False, collate_fn=collate_fn)
counter = 0
for (x, x_len), (y, y_len), nb_ops in loader:
    print(f"Iteration {counter}")
    print("Batched Input:")
    print(x, x_len)
    print("Batched Labels:")
    print(y, y_len)
    print("Batched Lengths:")
    print(nb_ops)
    print("")
    counter += 1
