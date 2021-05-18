import io
import torch
import random
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
        head = [next(f) for x in range(10000)]
        lines = [line.rstrip().split('|') for line in head]
        data = [xy.split('\t') for _, xy in lines]
        data = [xy for xy in data if len(xy) == 2]
    return data


def batch_sequences(x, y, env):
    """
    Take as input a list of n sequences (torch.LongTensor vectors) and return
    a tensor of size (slen, n) where slen is the length of the longest
    sentence, and a vector lengths containing the length of each sentence.
    """
    lengths_x = torch.LongTensor([len(s) + 2 for s in x])
    lengths_y = torch.LongTensor([len(s) + 2 for s in y])
    max_length = max(lengths_x.max().item(), lengths_y.max().item())
    sent_x = torch.LongTensor(
        max_length, lengths_x.size(0)).fill_(env.pad_index)
    sent_y = torch.LongTensor(
        max_length, lengths_y.size(0)).fill_(env.pad_index)
    assert lengths_x.min().item() > 2
    assert lengths_y.min().item() > 2

    sent_x[0] = env.eos_index
    for i, s in enumerate(x):
        sent_x[1:lengths_x[i] - 1, i].copy_(s)
        sent_x[lengths_x[i] - 1, i] = env.eos_index

    sent_y[0] = env.eos_index
    for i, s in enumerate(y):
        sent_y[1:lengths_y[i] - 1, i].copy_(s)
        sent_y[lengths_y[i] - 1, i] = env.eos_index

    return sent_x, sent_y, max_length


def collate_fn(elements):
    """
    Collate samples into a batch.
    """
    x, y = zip(*elements)
    nb_ops = [sum(int(word in env.OPERATORS) for word in seq) for seq in x]
    x = [torch.LongTensor([env.word2id[w]
                          for w in seq if w in env.word2id]) for seq in x]
    y = [torch.LongTensor([env.word2id[w]
                          for w in seq if w in env.word2id]) for seq in y]
    x, y, length = batch_sequences(x, y, env)
    return (x, length), (y, length), torch.LongTensor(nb_ops)


env = build_env(params)
path = 'sample_data/prim_fwd.txt'
data = read_data(path)
for i in range(len(data)):
    data[i] = tuple([sent.split(" ") for sent in data[i]])

# data[0] would be like :
# data[0]
# ["sub Y' pow x INT+ 2", 'mul div INT+ 1 INT+ 3 pow x INT+ 3']
loader = DataLoader(data, batch_size=25, shuffle=False, collate_fn=collate_fn)
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

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

gpt2 = GPT2Model.from_pretrained('gpt2')
in_layer = nn.Embedding(len(env.word2id), 768)
out_layer = nn.Linear(768, len(env.word2id))

for name, param in gpt2.named_parameters():
    # freeze all parameters except the layer norm and positional embeddings
    if 'ln' in name or 'wpe' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

parameters = list(gpt2.parameters()) + \
    list(in_layer.parameters()) + list(out_layer.parameters())
optimizer = torch.optim.Adam(parameters)
loss_fn = nn.CrossEntropyLoss()

for layer in (gpt2, in_layer, out_layer):
    layer.to(device=device)
    layer.train()


accuracies = list()
num_epoch = 10
for i in range(num_epoch):

  random.shuffle(data)
  
  for (x, x_len), (y, y_len), nb_ops in loader:

      x = x.to(device = device)
      y = y.to(device = device)

      embeddings = in_layer(x.reshape(x.shape[1], x.shape[0]))
      hidden_state = gpt2(inputs_embeds=embeddings).last_hidden_state[:,:]
      logits = out_layer(hidden_state)
      logits = logits.reshape(logits.shape[0], logits.shape[2], logits.shape[1])
      y = y.reshape(y.shape[1], y.shape[0])
      loss = loss_fn(logits, y)

      for i in range(logits.shape[0]):
        accuracies.append((logits[i,:,:].argmax(dim=0) == y[i, :]).float().mean().item())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


      if len(accuracies) % 1000 == 0:
          accuracy = sum(accuracies[-1000:]) / len(accuracies[-1000:])
          print(f'Samples: {len(accuracies)}, Accuracy: {accuracy}')

    
print(f'Final accuracy: {sum(accuracies[-1000:]) / len(accuracies[-1000:])}')
