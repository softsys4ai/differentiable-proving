from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, MBartTokenizer
from src.envs import build_env
import torch.nn.functional as F
import datasets
import random
import pandas as pd
from datasets import Dataset
import torch
import os
from datasets import load_metric
import io
import numpy as np
import sympy as sp
from src.utils import AttrDict
from gmp import one_shot_prune
from src.hf_utils import create_dataset_train, create_dataset_test
torch.cuda.empty_cache()

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
import argparse

parser = argparse.ArgumentParser(description='Differentiable Proving Evaluator')
parser.add_argument('-l', '--language', default='ro', help='SPECIFY LANGUAGE HERE')
parser.add_argument('-tr', '--train_file', default= 'data/train/prim_fwd_100k.train', help='SPECIFY PATH OF TRAIN DATA HERE')
parser.add_argument('-v', '--valid_file', default='data/valid/prim_fwd.valid', help='SPECIFY PATH OF VALID DATA HERE')
parser.add_argument('-t', '--test_file', default='prim_fwd_1k', help='SPECIFY PATH OF TEST DATA HERE')
parser.add_argument('-isen', '--is_source_en', help='IS SOURCE LANGUAGE ENGLISH?', action='store_true')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(device)
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
    'tasks': 'prim_bwd',
    'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1',
})

language = args.language # SPECIFY LANGUAGE HERE.
env = build_env(params)
path1 = args.train_file    # SPECIFY PATH OF TRAINING DATA HERE.
train_dataset = create_dataset_train(path=path1, count=100000, language = language)
path2 = args.valid_file    # SPECIFY PATH OF VALIDATION DATA HERE. WE WILL USE ALL OF VALIDATION DATA, NO NEED TO SPECIFY COUNT.
valid_dataset = create_dataset_test(path=path2, language= language)

"""# Tokenizing the Data"""
Model_Type = 'mbart'

if Model_Type == 'mbart':
    model_checkpoint = "facebook/mbart-large-en-{}".format(language) # SPECIFY PRE-TRAINED MODEL HERE. 
    metric = load_metric("sacrebleu")
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO")
elif Model_Type == 'Marian':
    if args.is_source_en:
        print('Source Language is English')
        model_checkpoint = "Helsinki-NLP/opus-mt-en-{}".format(language)
    else:
        print('Source Language is {}'.format(language))
        model_checkpoint = "Helsinki-NLP/opus-mt-{}-en".format(language)
    metric = load_metric("sacrebleu")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "not important."
else:
    prefix = ""

"""# Create the Final Data Set"""

datasetM = {'train': train_dataset,
            'validation': valid_dataset}
max_input_length = 1024 # Set to 512 if it is Marian-MT
max_target_length = 1024 # Set to 512 if it is Marian-MT
source_lang = "en"
target_lang = language

tokenized_datasets_train = datasetM['train'].map(preprocess_function_new, batched=True, num_proc = 48)
tokenized_datasets_valid = datasetM['validation'].map(preprocess_function_new, batched=True)

"""#  Fine-tuning the model"""

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# sparcify model to 50%
model = one_shot_prune(model, 0.5)
model.to('cuda')
# freeze all layers except the norm layer
# for name, param in model.named_parameters():
#     if 'norm' or 'attn' in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

if args.is_source_en:
    name = '{}_{}_{}_{}'.format(args.train_file, args.test_file, 'en', language)
else:
    name = '{}_{}_{}_{}'.format(args.train_file,args.test_file, language, 'en')
    


batch_size = 32
args = Seq2SeqTrainingArguments(
    "test-translation_{}".format(name),
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=15,
    predict_with_generate=False,
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_valid,
    data_collator=data_collator,
    tokenizer=tokenizer
)


trainer.train()
model_name = name # SPECIFY MODEL SAVING NAME HERE.
torch.save(model, 'checkpoints/{}'.format(model_name))