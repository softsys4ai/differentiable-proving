from src.envs.sympy_utils import simplify
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from src.envs import build_env
from torch.utils.data import DataLoader
from functools import partial
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import datasets
import random
import pandas as pd
from IPython.display import display, HTML
from datasets import Dataset
import pandas as pd
from logging import getLogger
import torch
import os
from datasets import load_dataset, load_metric
import csv
import io
import numpy as np
import sympy as sp
import torch
import random
import sys
from src.utils import AttrDict
from datasets import load_dataset, load_metric
import sentencepiece
from transformers.models.bert.modeling_bert import BertLayer

# Required Functions


def preprocess_function_new(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def read_data(path, number_of_samples):
    with io.open(path, mode='r', encoding='utf-8') as f:
        head = [next(f) for x in range(number_of_samples)]
        lines = [line.rstrip().split('|') for line in head]
        data = [xy.split('\t') for _, xy in lines]
        data = [xy for xy in data if len(xy) == 2]
    return data


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def convert_to_sympy(s):
    tok = s.split()
    hyp = env.prefix_to_infix(tok)
    hyp = env.infix_to_sympy(hyp)
    return hyp


def create_dataset(path, count):
    data = read_data(path, count)
    text = []
    label = []
    for i in range(len(data)):
        text.append(data[i][0])
        label.append(data[i][1])
    raw_datasets = [{'en': text[i], 'ro': label[i]}
                    for i in range(len(text))]

    raw_datasets_t = {}
    for i in range(len(raw_datasets)):
        raw_datasets_t.setdefault('translation', []).append(
            {'translation': raw_datasets[i]})

    df = pd.DataFrame.from_dict(raw_datasets_t['translation'])
    dataset = Dataset.from_pandas(df)
    return dataset


def prediction(data, model, tokenizer, number_of_samples):
    count_acc = 0
    count = 0
    for i in range(number_of_samples):
        text = data['translation'][i]['en']
        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids.to(device='cuda'))
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        actual = data['translation'][i]['ro']
        try:
            actual_s = convert_to_sympy(actual)
            decoded_s = convert_to_sympy(decoded)
            res = "OK" if simplify(decoded_s - actual_s,
                                   seconds=1) == 0 else "NO"
            if res == 'OK':
                count_acc += 1
        except:
            with open('invalids.txt', 'a') as f:
                f.write(actual)
                f.write('\n')
                f.write(decoded)
                f.write('\n')
            count += 1
            continue
    print("Train Accuracy:", count_acc/number_of_samples)
    print(count)


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

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
path1 = "sample_data/prim_fwd.train"
train_dataset = create_dataset(path=path1, count=10000)
path2 = "sample_data/prim_fwd.valid"
valid_dataset = create_dataset(path=path2, count=1000)
path3 = "sample_data/prim_fwd.test"
test_dataset = create_dataset(path=path3, count=500)

"""# Tokenizing the Data"""
model_checkpoint = "Helsinki-NLP/opus-mt-en-ro"
metric = load_metric("sacrebleu")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

if "mbart" in model_checkpoint:
    tokenizer.src_lang = "en-XX"
    tokenizer.tgt_lang = "ro-RO"
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "translate English to Romanian: "
else:
    prefix = ""

"""# Create the Final Data Set"""

datasetM = {'train': train_dataset,
            'validation': valid_dataset, 'test': test_dataset}
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "ro"

tokenized_datasets_train = datasetM['train'].map(
    preprocess_function_new, batched=True)
tokenized_datasets_valid = datasetM['validation'].map(
    preprocess_function_new, batched=True)
tokenized_datasets_test = datasetM['test'].map(
    preprocess_function_new, batched=True)

"""#  Fine-tuning the model"""
torch.cuda.empty_cache()
model = torch.load('models/model_1000')

prediction(data=tokenized_datasets_test, model=model,
           tokenizer=tokenizer, number_of_samples=500)
