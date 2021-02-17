# coding: utf-8
"""
Implementation of normalizers and tokenizers
"""


def default_normalizer(sentence: str):
    return sentence


def default_tokenizer(sentence: str):
    return sentence.split(' ')


NORMALIZERS = {'default': default_normalizer}

TOKENIZERS = {'default': default_tokenizer}
