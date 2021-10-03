from datasets import Dataset
import pandas as pd
import io
from src.envs.sympy_utils import simplify
from enum import Enum
import random


def read_data_train(path, number_of_samples):
    with io.open(path, mode='r', encoding='utf-8') as f:
        head = [next(f) for x in range(number_of_samples)]
        lines = [line.rstrip().split('|') for line in head]
        data = [xy.split('\t') for _, xy in lines]
        data = [xy for xy in data if len(xy) == 2]
    return data


def read_data_test(path):
    with io.open(path, mode='r', encoding='utf-8') as f:
        lines = [line.rstrip().split('|') for line in f]
        data = [xy.split('\t') for _, xy in lines]
        data = [xy for xy in data if len(xy) == 2]
    return data

def read_data_test_new(path):
    with io.open(path, mode='r', encoding='utf-8') as f:
        lines = [line.rstrip() for line in f]
        data = [xy.split('\t') for xy in lines]
        data = [xy for xy in data if len(xy) == 2]
    return data

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def convert_to_sympy(s, env):
    tok = s.split()
    hyp = env.prefix_to_infix(tok)
    hyp = env.infix_to_sympy(hyp)
    return hyp


def create_dataset_train(path, count, language):
    data = read_data_train(path, count)
    text = []
    label = []
    for i in range(len(data)):
        text.append(data[i][0])
        label.append(data[i][1])
    raw_datasets = [{'en': text[i], language: label[i]}
                    for i in range(len(text))]

    raw_datasets_t = {}
    for i in range(len(raw_datasets)):
        raw_datasets_t.setdefault('translation', []).append({'translation': raw_datasets[i]})

    df = pd.DataFrame.from_dict(raw_datasets_t['translation'])
    dataset = Dataset.from_pandas(df)
    return dataset


def create_dataset_test(path, language):
    data = read_data_test(path)
    text = []
    label = []
    for i in range(len(data)):
        text.append(data[i][0])
        label.append(data[i][1])
    raw_datasets = [{'en': text[i], language: label[i]}
                    for i in range(len(text))]

    raw_datasets_t = {}
    for i in range(len(raw_datasets)):
        raw_datasets_t.setdefault('translation', []).append({'translation': raw_datasets[i]})

    df = pd.DataFrame.from_dict(raw_datasets_t['translation'])
    dataset = Dataset.from_pandas(df)
    return dataset

def create_dataset_test_new(path, language):
    data = read_data_test_new(path)
    text = []
    label = []
    for i in range(len(data)):
        text.append(data[i][0])
        label.append(data[i][1])
    raw_datasets = [{'en': text[i], language: label[i]}
                    for i in range(len(text))]

    raw_datasets_t = {}
    for i in range(len(raw_datasets)):
        raw_datasets_t.setdefault('translation', []).append({'translation': raw_datasets[i]})

    df = pd.DataFrame.from_dict(raw_datasets_t['translation'])
    dataset = Dataset.from_pandas(df)
    return dataset


evaluationType = Enum('evaluationType', 'Training Validation Test')


def evaluation_function(totalNumberOfEvaluation, tokenized_datasets, evalType, tokenizer, model, batch_size, env, num_beams, language):
    count_trueEstimation = 0
    count_nonMathExpressionEstimation = 0
    numberOfBatches = int(totalNumberOfEvaluation / batch_size)
    for j_batchIndex in range(numberOfBatches):
        text = [tokenized_datasets['translation'][i]['en'] for i in range(
            j_batchIndex * batch_size, (j_batchIndex+1) * batch_size)]
        input_batch = tokenizer(text, return_tensors="pt", padding=True)
        try:
            outputs = model.generate(**input_batch.to(device='cuda'), num_beams = num_beams)
            decoded_batch = [tokenizer.decode(
                t, skip_special_tokens=True) for t in outputs]
            for k_indexInsideBatch in range(batch_size):
                decoded = decoded_batch[k_indexInsideBatch]
                ii_indexInWhole = j_batchIndex * batch_size + k_indexInsideBatch
                actual = tokenized_datasets['translation'][ii_indexInWhole][language]
                try:
                    actual_s = convert_to_sympy(actual, env)
                    decoded_s = convert_to_sympy(decoded, env)
                    res = True if simplify(
                        decoded_s - actual_s, seconds=1) == 0 else False
                    if res == True:
                        count_trueEstimation += 1
                except:
                    count_nonMathExpressionEstimation += 1
                    continue
        except:
            totalNumberOfEvaluation -= 1
            continue
    print(evalType.name, "Accuracy:", 100 * count_trueEstimation/totalNumberOfEvaluation)
    print("NumberOfFalseEstimation", count_nonMathExpressionEstimation)

