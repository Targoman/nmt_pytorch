# coding: utf-8
"""
Definition of global constants
"""

import os
import torch
import numpy as np
import random

import logging
import inspect
import typing
from typing import Callable, Union, T, List, Tuple
from nmt.config import Configuration

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

epsilon = 1e-6

configuration = Configuration()

class IgnoreMeta(type):
    def __getitem__(cls, val):
        result = cls()
        result.__args__ = val
        return result

class Ignore(object, metaclass=IgnoreMeta):
    pass

def configured(parent_configuration: str = None):

    global configuration
    config_provider = configuration
    if parent_configuration is not None:
        path = parent_configuration.split('.')
        for part in path:
            config_provider = config_provider.ensure_submodule(part)

    def register_params(parameter_specs):

        ACCEPTABLE_TYPES = [
            int, bool, float, str, List[int], List[bool], List[float],
            List[str], Tuple[int], Tuple[bool], Tuple[float], Tuple[str]
        ]

        param_names = []

        for param_name in parameter_specs:
            param_spec = parameter_specs[param_name]
            if param_spec.annotation not in ACCEPTABLE_TYPES:
                continue
            param_names.append(param_name)
            config_provider.ensure_param(
                param_name,
                None \
                if param_spec.default == inspect.Parameter.empty else \
                param_spec.default
            )

        return param_names

    def make_configured_function(f: Callable):

        signature = inspect.signature(f)
        param_names = register_params(signature.parameters)

        def configured_f(*args, **kwargs):
            params = {
                name: getattr(config_provider, name)
                for name in param_names
            }
            params.update(kwargs)
            return f(*args, **params)

        configured_f.__name__ = f.__name__

        return configured_f

    def make_configured_class(c: T):

        signature = inspect.signature(c.__init__)
        param_names = register_params(signature.parameters)

        class configured_c(c):
            def __init__(self, *args, **kwargs):
                params = {
                    name: getattr(config_provider, name)
                    for name in param_names
                }
                params.update(kwargs)
                c.__init__(self, *args, **params)

        configured_c.__name__ = c.__name__

        return configured_c

    def make_configured(o: Union[Callable, object]):
        if inspect.isclass(o):
            return make_configured_class(o)
        else:
            return make_configured_function(o)

    return make_configured


logger: logging.Logger = None

xm = None
if ('COLAB_TPU_ADDR' in  os.environ) and os.environ['COLAB_TPU_ADDR']:
    import torch_xla
    import torch_xla.core.xla_model
    xm = torch_xla.core.xla_model

def mark_optimization_step():
    if xm is not None:
        xm.mark_step()

device = None
def set_device(value):
    global device
    device = value

@configured('train')
def get_device(use_gpu: bool = False):
    global device
    if device is None:
        if use_gpu:
            if xm is not None:
                return xm.xla_device()
            if torch.cuda.is_available():
                return torch.device('cuda')
        return torch.device('cpu')
    return device

def set_random_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    if xm is not None:
        xm.set_rng_state(seed)
    np.random.seed(seed)
    random.seed(seed)

@configured('model')
def make_logger(output_path: str = './results/'):
    global logger

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(message)s')

    fh = logging.FileHandler('{}/train.log'.format(output_path))
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

def get_logger():
    global logger
    
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s %(message)s')

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger
