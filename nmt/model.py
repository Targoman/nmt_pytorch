# coding: utf-8
"""
Translation models
"""

import os
import math
import sys
from importlib import util as iu

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nmt.dataset import Vocabulary
from nmt.common import configured
from nmt.encoderdecoder import EncoderDecoder

def load_model_module(type: str):
    if not f'nmt.{type}' in sys.modules:
        module_spec = iu.spec_from_file_location(f'nmt.{type}', f'{os.path.dirname(__file__)}/models/{type}.py')
        if module_spec is None:
            raise Exception(
                f'No `{type}` exists in defined seq2seq models.'
            )
        model_module = iu.module_from_spec(module_spec)
        module_spec.loader.exec_module(model_module)
        sys.modules[f'nmt.{type}'] = model_module
    else:
        model_module = sys.modules[f'nmt.{type}']

    return model_module

@configured('model')
def build_model(
    src_vocab: Vocabulary, tgt_vocab: Vocabulary, type: str = 'transformer'
):
    model_module = load_model_module(type)
    return model_module.Model(src_vocab, tgt_vocab)

@configured('model')
def get_model_short_description(type: str = 'transformer'):
    return 'transformer_baseline'

    # model_module = load_model_module(type)
    # if hasattr(model_module.Model, 'short_description'):
    #     return model_module.Model.short_description()
    
    #return None

@configured('model')
def get_model_source_code_path(type: str = 'transformer'):
    return f'{os.path.dirname(__file__)}/models/{type}.py'