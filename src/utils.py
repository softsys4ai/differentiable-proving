import os
import re
import sys
import math
import time
import pickle
import random
import argparse
import subprocess

import errno
import signal
from functools import wraps, partial

from .logger import create_logger


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


