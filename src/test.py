import io
import torch

from src.utils import AttrDict
from src.envs import build_env
from src.model import build_modules

from src.utils import to_cuda
from src.envs.sympy_utils import simplify

from torch.utils.data import DataLoader

from transformers.models.gpt2.modeling_gpt2 import GPT2Model

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

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


def predict(tensor, pred_mask, y, get_scores, emb_dim):
    """
    Given the last hidden state, compute word scores and/or the loss.
        `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
            we need to predict a word
        `y` is a LongTensor of shape (pred_mask.sum(),)
        `get_scores` is a boolean specifying whether we need to return scores
    """
    x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, emb_dim)
    assert (y == env.pad_index).sum().item() == 0
    scores = proj(x).view(-1, len(env.id2word))
    loss = F.cross_entropy(scores, y, reduction='mean')
    return scores, loss


def train_batch(x1, len1, x2, len2, params, emb_dim, optimizer):
    # target words to predict
    alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
    pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
    y = x2[1:].masked_select(pred_mask[:-1])
    assert len(y) == (len2 - 1).sum().item()

    embeddings = in_layer(x.reshape(1, -1))
    out = gpt2(inputs_embeds=embeddings)
    _, loss = predict(tensor=out, pred_mask=pred_mask, y=y, get_scores=False, emb_dim=emb_dim)
    print(loss.item())

    optimizer.zero_grad()
    # Calculate the gradients
    loss.backward()
    # Update the parameters
    optimizer.step()


env = build_env(params)
path = 'C:\\Users\\Kimia\\Desktop\\DeskTop\\5 Term\\Research UCS\\transformer\\symbolic math\\Code\\differentiable-proving\\sample_data\\prim_fwd.txt'
data = read_data(path)
for i in range(len(data)):
    data[i] = tuple([sent.split(" ") for sent in data[i]])

# data[0] would be like :
# data[0]
# ["sub Y' pow x INT+ 2", 'mul div INT+ 1 INT+ 3 pow x INT+ 3']
loader = DataLoader(data, batch_size=32, shuffle=False, collate_fn=collate_fn)
# loader.dataset
# Go through one loop
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
    break

print('batched input shape:', x.shape)
print('batched output shape:', y.shape)
# each column is showing one training example

gpt2 = GPT2Model.from_pretrained('gpt2')
in_layer = nn.Embedding(1, 768)

for name, param in gpt2.named_parameters():
    # freeze all parameters except the layer norm and positional embeddings
    if 'ln' in name or 'wpe' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

parameters = list(gpt2.parameters()) + list(in_layer.parameters())

for layer in (gpt2, in_layer):
    layer.train()

emb_dim = 768
proj = nn.Linear(emb_dim, params.n_words, bias=True)
optimizer = torch.optim.Adam(parameters)

for (x, x_len), (y, y_len), nb_ops in loader:
    train_batch(x, x_len, y, y_len, params, emb_dim, optimizer)
