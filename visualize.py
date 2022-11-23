# histogram of the huggung face model based on layer type
import torch 
import argparse
import numpy as np
from functools import reduce
import operator
import matplotlib.pyplot as plt
from transformers import AutoModelForSeq2SeqLM

def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)

def get_weights(model, layer_number):
    weight_attntion  = []
    weight_fc = []
    weight_norm = []
    for module_name, _ in model.named_modules():
        if 'attn' in module_name and layer_number in module_name:
            try:
                l = get_module_by_name(model, module_name).weight
                l = torch.flatten(l).tolist()
                weight_attntion.extend(l)
            except:
                continue
        elif 'fc' in module_name and layer_number in module_name:
            try:
                l = get_module_by_name(model, module_name).weight
                l = torch.flatten(l).tolist()
                weight_fc.extend(l)
            except:
                continue
        elif 'norm' in module_name and layer_number in module_name:
            try:
                l = get_module_by_name(model, module_name).weight
                l = torch.flatten(l).tolist()
                weight_norm.extend(l)
            except:
                continue
    weight_attntion = np.array(weight_attntion)
    weight_fc = np.array(weight_fc)
    weight_norm = np.array(weight_norm)
    return weight_attntion, weight_fc, weight_norm
        
parser = argparse.ArgumentParser(description='Weight visualization')
parser.add_argument('-m', '--model', default='mbart_prim_ibp_1M_en_ro', help='Model path')
args = parser.parse_args()
saved_path = 'models/{}'.format(args.model)
model = torch.load(saved_path)

model_checkpoint = "facebook/mbart-large-en-{}".format('ro')
model_pretrained = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

layers = ['.0.', '.1.', '.2.', '.3.', '.4.', '.5.', '.6.', '.7.', '.8.', '.9.', '.10.', '.11.']
for i, layer_number in enumerate(layers):
    w_attn_pretrained, w_fc_pretrained, w_norm_pretrained = get_weights(model_pretrained, layer_number)
    w_attn_finetuned, w_fc_finetuned, w_norm_finetuned = get_weights(model, layer_number)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams['font.size'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams["figure.figsize"] = [16, 10]
    fig, axs = plt.subplots(3)
    axs[0].hist(w_attn_pretrained, bins = 'auto', label = 'pre-trained')
    axs[0].hist(w_attn_finetuned, bins = 'auto', label = 'fine-tuned')
    axs[0].set_title('Attention weights')
    axs[1].hist(w_fc_pretrained, bins = 'auto', label = 'pre-trained')
    axs[1].hist(w_fc_finetuned, bins = 'auto', label = 'fine-tuned')
    axs[1].set_title('FC weights')
    axs[2].hist(w_norm_pretrained, bins = 'auto', label = 'pre-trained')
    axs[2].hist(w_norm_finetuned, bins = 'auto', label = 'fine-tuned')
    axs[2].set_title('Batch Norm weights')
    plt.legend()
    plt.savefig('visual/{}_{}.png'.format(args.model, i))