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
from datasets import load_dataset, load_metric
import io
import numpy as np
import sympy as sp
from src.utils import AttrDict 
from src.hf_utils import evaluation_function, create_dataset_test, postprocess_text, ensemble_evaluation
from enum import Enum
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

import argparse

parser = argparse.ArgumentParser(description='Differentiable Proving Evaluator')
parser.add_argument('-l', '--language', default='ro', help='SPECIFY LANGUAGE HERE')
parser.add_argument('-t', '--test_file', default='prim_bwd_1k', help='SPECIFY PATH OF TEST DATA HERE')
parser.add_argument('-m', '--model', default='prim_bwd_en_ro_100k_wrong_sin_cos', help='SPECIFY PRE-TRAINED MODEL HERE')
parser.add_argument('-isen', '--is_source_en', help='IS SOURCE LANGUAGE ENGLISH?', action='store_true')
args = parser.parse_args()
    
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
    'tasks': 'prim_bwd',
    'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1',
})

language = args.language # SPECIFY LANGUAGE HERE.
env = build_env(params)

"""# Tokenizing the Data"""
Model_Type = 'Marian'
# is_source_en = args.is_source_en # IS SOURCE LANGUAGE ENGLISH?

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


max_input_length = 1024 # Set to 512 if it is Marian-MT
max_target_length = 1024 # Set to 512 if it is Marian-MT
source_lang = "en"
target_lang = language


      
torch.cuda.empty_cache()
path = "../SymbolicMathematics/new_test/{}.test".format(args.test_file) # SPECIFY PATH OF TEST DATA HERE.
test_dataset = create_dataset_test(path=path, language= language)  
datasetM = {'test': test_dataset}
tokenized_datasets_test = datasetM['test'].map(preprocess_function_new, batched=True)
saved_path = 'checkpoints/{}'.format(args.model)
model = torch.load(saved_path)  # SPECIFY LOADING PATH HERE.
evaluationType = Enum('evaluationType', 'Training Validation Test')
batch_size = 16
predictions = ensemble_evaluation(1000, tokenized_datasets_test, evaluationType.Test, tokenizer, model, batch_size, env, num_beams= 1, language= language)
predictions = np.array(predictions)
if args.is_source_en:
    np.save('predictions/en_{}_{}.npy'.format(args.language, args.test_file), predictions)
else:
    np.save('predictions/{}_en_{}.npy'.format(args.language, args.test_file), predictions)

print(len(predictions))
print(100 * sum(predictions) / len(predictions))
