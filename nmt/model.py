# coding: utf-8
"""
Translation models
"""

import os
import math
import sys
import shutil
from importlib import util as iu

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nmt.dataset import Vocabulary
from nmt.common import get_configuration, configuration_stack, configured
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
    model_module = load_model_module(type)
    if hasattr(model_module.Model, 'short_description'):
        return model_module.Model.short_description()
    return None

@configured('model')
def get_model_source_code_path(type: str = 'transformer'):
    return f'{os.path.dirname(__file__)}/models/{type}.py'

def update_and_ensure_model_output_path(mode, index):
    model_short_description = get_model_short_description()
    model_source_code_path = get_model_source_code_path()
    model_configuration = get_configuration().ensure_submodule('model')

    base_output_path = model_configuration.output_path

    if model_short_description is not None:
        base_output_path = os.path.join(model_configuration.output_path, f'{model_short_description}')

    if mode != 'train':
        if index is None:
            index = 1
            while not os.path.exists(f'{base_output_path}/{index:03}'):
                index += 1
                if index > 1e4:
                    raise ValueError(f'No model could be found in {base_output_path}.')
        model_configuration.output_path = f'{base_output_path}/{index:03}'
    else:
        index = 1
        while os.path.exists(f'{base_output_path}/{index:03}'):
            index += 1
            if index > 1e4:
                raise ValueError(f'No model could be found in {base_output_path}.')
        model_configuration.output_path = f'{base_output_path}/{index:03}'
        os.makedirs(model_configuration.output_path, exist_ok=True)
        shutil.copyfile(model_source_code_path, os.path.join(model_configuration.output_path, 'model.py'))
