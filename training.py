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

# Set up and Loading the Data


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
path = "sample_data/prim_fwd.train"
data = read_data(path, 100000)

"""# Pre-Processing the Data
Here we pre-process our data, so that it matches the format in our reference code: [Hugging Face Translation Task Example](https://github.com/huggingface/notebooks/blob/master/examples/translation.ipynb)
# Training Dataset
"""
train_text = []
train_label = []
for i in range(len(data)):
    train_text.append(data[i][0])
    train_label.append(data[i][1])
raw_datasets = [{'en': train_text[i], 'ro': train_label[i]}
                for i in range(len(train_text))]

raw_datasets_train = {}
for i in range(len(raw_datasets)):
    raw_datasets_train.setdefault('translation', []).append(
        {'translation': raw_datasets[i]})

df2 = pd.DataFrame.from_dict(raw_datasets_train['translation'])
train_dataset = Dataset.from_pandas(df2)

"""## Validation Dataset"""

path2 = "sample_data/prim_fwd.valid"
data2 = read_data(path2, 9000)
valid_text = []
valid_label = []
for i in range(len(data2)):
    valid_text.append(data2[i][0])
    valid_label.append(data2[i][1])
raw_datasets1 = [{'en': valid_text[i], 'ro': valid_label[i]}
                 for i in range(len(valid_text))]

raw_datasets_vlaid = {}
for i in range(len(raw_datasets1)):
    raw_datasets_vlaid.setdefault('translation', []).append(
        {'translation': raw_datasets1[i]})

df_valid = pd.DataFrame.from_dict(raw_datasets_vlaid['translation'])
valid_dataset = Dataset.from_pandas(df_valid)

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
            'validation': valid_dataset}
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "ro"

tokenized_datasets_train = datasetM['train'].map(
    preprocess_function_new, batched=True)
tokenized_datasets_valid = datasetM['validation'].map(
    preprocess_function_new, batched=True)

"""#  Fine-tuning the model"""

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 25
args = Seq2SeqTrainingArguments(
    "test-translation",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=25,
    predict_with_generate=True,
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
 
trainer.train()
torch.save(model, 'models/model3')
