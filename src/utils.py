import os
import re
import sys
import math
import time
import pickle
import random
import argparse
import subprocess
from datasets import Dataset
import pandas as pd
import io
from src.envs.sympy_utils import simplify
from enum import Enum

import errno
import signal
from functools import wraps, partial

from .logger import create_logger


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

CUDA = True


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(
        params.dump_path, 'params.pkl'), 'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(
        params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == '':
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    if not CUDA:
        return args
    return [None if x is None else x.cuda() for x in args]


class TimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):

    def decorator(func):

        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(
                _handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(
                signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator


def read_data_train(path, number_of_samples):
    with io.open(path, mode='r', encoding='utf-8') as f:
        head = [next(f) for x in range(number_of_samples)]
        lines = [line.rstrip().split('|') for line in head]
        data = [xy.split('\t') for _, xy in lines]
        data = [xy for xy in data if len(xy) == 2]
    return data


def read_data_test(path, number_of_samples):
    with io.open(path, mode='r', encoding='utf-8') as f:
        lines = [line.rstrip().split('|') for line in f]
        lines = random.sample(lines, number_of_samples)
        data = [xy.split('\t') for _, xy in lines]
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


def create_dataset_train(path, count):
    data = read_data_train(path, count)
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


def create_dataset_test(path, count):
    data = read_data_test(path, count)
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


evaluationType = Enum('evaluationType', 'Training Validation Test')


def evaluation_function(totalNumberOfEvaluation, tokenized_datasets, evalType, tokenizer, model, batch_size, env):
    count_trueEstimation = 0
    count_nonMathExpressionEstimation = 0
    numberOfBatches = int(totalNumberOfEvaluation / batch_size)
    for j_batchIndex in range(numberOfBatches):
        text = [tokenized_datasets['translation'][i]['en'] for i in range(
            j_batchIndex * batch_size, (j_batchIndex+1) * batch_size)]
        input_batch = tokenizer(text, return_tensors="pt", padding=True)
        outputs = model.generate(**input_batch.to(device='cuda'))
        decoded_batch = [tokenizer.decode(
            t, skip_special_tokens=True) for t in outputs]
        for k_indexInsideBatch in range(batch_size):
            decoded = decoded_batch[k_indexInsideBatch]
            ii_indexInWhole = j_batchIndex * batch_size + k_indexInsideBatch
            actual = tokenized_datasets['translation'][ii_indexInWhole]['ro']
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
    print(evalType.name, "Accuracy:", 100 *
          count_trueEstimation/totalNumberOfEvaluation)
    print("NumberOfFalseEstimation", count_nonMathExpressionEstimation)
