# coding: utf-8
"""
Implementation of a simple yet fast dataset for neural machine translation
"""

import os
import pickle
import random
from typing import List, Callable, IO, Union, Any

# pylint: disable=no-member,no-value-for-parameter
import torch

from nmt.common import UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN, configured, get_logger
from nmt.preprocess import NORMALIZERS, TOKENIZERS


class Vocabulary(object):
    SPECIALS = [
        (UNK_TOKEN, 'unk_index'), (PAD_TOKEN, 'pad_index'),
        (SOS_TOKEN, 'sos_index'), (EOS_TOKEN, 'eos_index')
    ]

    def __init__(self):
        self.stoi = {}
        self.stof = {}
        self.itos = []

        for token, token_index_name in self.SPECIALS:
            self.__setattr__(token_index_name, self.append_and_get_index(token))

    def __update_special_indexes(self):
        for token, token_index_name in self.SPECIALS:
            self.__setattr__(token_index_name, self.get_index(token))

    def __getstate__(self):
        return [self.stoi, self.stof, self.itos]

    def __setstate__(self, state):
        self.stoi, self.stof, self.itos = state
        self.__update_special_indexes()

    def append_and_get_index(self, token: str):
        if token == '':
            raise Exception()
        if not token in self.stoi:
            index = len(self.itos)
            self.stoi[token] = index
            self.stof[token] = 1
            self.itos.append(token)
            return index
        self.stof[token] += 1
        return self.stoi[token]

    def get_index(self, token: str):
        if not token in self.stoi:
            return 0  # self.stoi[UNK_TOKEN]
        return self.stoi[token]

    def get_word(self, index: int):
        assert index < len(
            self.itos
        ), 'Word index is out of vocabulary size ({} > {})'.format(
            index, len(self.itos)
        )
        return self.itos[index]

    def save_as_csv(self, output_path: str):
        sorted_tokens = sorted(
            self.itos[4:], key=lambda e: self.stof[e], reverse=True
        )
        with open(output_path, 'w') as f:
            for token in [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
                f.write(
                    '{}\t{}\t{}\n'.format(
                        token, self.stoi[token], self.stof[token]
                    )
                )
            for token in sorted_tokens:
                f.write(
                    '{}\t{}\t{}\n'.format(
                        token, self.stoi[token], self.stof[token]
                    )
                )

    def load_from_csv(self, input_path: str):
        with open(input_path) as f:
            for l in f:
                parts = l.strip().split('\t')
                self.stoi[parts[0]] = int(parts[1])
                self.stof[parts[0]] = int(parts[2])
        self.itos = sorted(self.stoi.keys(), key=lambda e: self.stoi[e])
        self.__update_special_indexes()
        return self


class Field(object):
    def __init__(
        self,
        lowercase: bool = True,
        vocabulary: Vocabulary = None,
        normalize: str = None,
        tokenize: str = None,
        force_vocabulary_update: bool = False
    ):

        self.lowercase = lowercase
        self.has_external_vocabulary = vocabulary is not None
        self.force_vocabulary_update = force_vocabulary_update
        self.vocabulary = vocabulary if self.has_external_vocabulary else Vocabulary(
        )
        self.normalizer_name = 'default' if normalize is None else normalize
        self.normalize = NORMALIZERS[self.normalizer_name]
        self.tokenizer_name = 'default' if tokenize is None else tokenize
        self.tokenize = TOKENIZERS[self.tokenizer_name]

    def to_word_indexes(self, sentence: str):

        if self.lowercase:
            result = sentence.strip().lower()
        else:
            result = sentence.strip()
        result = self.tokenize(self.normalize(result))
        if not self.force_vocabulary_update:
            result = [self.vocabulary.sos_index
                     ] + [self.vocabulary.get_index(e)
                          for e in result] + [self.vocabulary.eos_index]
        else:
            result = [self.vocabulary.sos_index] + [
                self.vocabulary.append_and_get_index(e) for e in result
            ] + [self.vocabulary.eos_index]
        return result

    def to_sentence_str(self, word_indexes: List[int]):
        return ' '.join(
            self.vocabulary.itos[e] for e in word_indexes if e not in [
                self.vocabulary.sos_index, self.vocabulary.eos_index,
                self.vocabulary.pad_index
            ]
        )

    def __getstate__(self):
        if self.has_external_vocabulary:
            return [
                self.lowercase, True, self.normalizer_name, self.tokenizer_name
            ]
        else:
            return [
                self.lowercase, False, self.vocabulary, self.normalizer_name,
                self.tokenizer_name
            ]

    def __setstate__(self, state):
        if state[1]:
            self.lowercase = state[0]
            self.has_external_vocabulary = True
            self.vocabulary = None
            self.normalizer_name = state[2]
            self.normalize = NORMALIZERS[state[2]]
            self.tokenizer_name = state[3]
            self.tokenize = TOKENIZERS[state[3]]
        else:
            self.lowercase = state[0]
            self.has_external_vocabulary = False
            self.vocabulary = state[2]
            self.normalizer_name = state[3]
            self.normalize = NORMALIZERS[state[3]]
            self.tokenizer_name = state[4]
            self.tokenize = TOKENIZERS[state[4]]


class Corpora(object):
    def __init__(self, fields: List[Field]):
        self.fields = fields
        self.samples = []
        self.__random_state = None

    def append(self, sentences: List[str]):
        assert len(self.fields) == len(sentences), \
            'Number of sentences and fields do not match. ({} != {})'.format(
            len(sentences), len(self.fields))

        sample = [f.to_word_indexes(s) for f, s in zip(self.fields, sentences)]

        self.samples.append(sample)

    def __getstate__(self):
        external_vocabularies = list(
            set(f.vocabulary for f in self.fields if f.has_external_vocabulary)
        )
        field_vocabulary_index = [
            (
                external_vocabularies.index(f.vocabulary)
                if f.has_external_vocabulary else -1
            ) for f in self.fields
        ]
        return [
            self.samples, self.fields, external_vocabularies,
            field_vocabulary_index, self.__random_state
        ]

    def __setstate__(self, state):
        self.samples = state[0]
        self.fields = state[1]
        for field, vocabulary_index in zip(state[1], state[3]):
            if vocabulary_index < 0:
                continue
            field.vocabulary = state[2][vocabulary_index]
        self.__random_state = state[4] if len(state) >= 5 else None

    def save(self, output: Union[str, IO[Any]]):
        if isinstance(output, str):
            stream = open(output, 'wb')
        else:
            stream = output
        pickle.dump(self, stream)

    @staticmethod
    def load(input: Union[str, IO[Any]]):
        if isinstance(input, str):
            stream = open(input, 'rb')
        else:
            stream = input
        corpora = pickle.load(stream)
        if not isinstance(corpora, Corpora):
            raise Exception(
                'Corpora::load: Input to load function is not an instance of `Corpora`.'
            )
        return corpora

    def iterate(
        self,
        device: Any,
        batch_size_limit: int,
        batch_limit_by_tokens: bool,
        sort_by_length: bool = False,
        shuffle: bool = True
    ):
        n_fields = len(self.fields)
        pad_indexes = [f.vocabulary.pad_index for f in self.fields]

        def get_torch_batch(raw_batch):

            max_lengths = [
                max(len(sample[i]) for sample in raw_batch)
                for i in range(n_fields)
            ]

            def pad_sequence(s, pad_index, max_length):
                return s + [pad_index for _ in range(max_length - len(s))]

            return [
                torch.LongTensor(
                    [
                        pad_sequence(sample[i], pad_indexes[i], max_lengths[i])
                        for sample in raw_batch
                    ]
                ).to(device) for i in range(n_fields)
            ]

        def get_batch_size_in_tokens(raw_batch):
            max_seq_length = max(
                max(len(sample[i]) for sample in raw_batch)
                for i in range(n_fields)
            )
            return max_seq_length * len(raw_batch)

        def get_batch_size_in_samples(raw_batch):
            return len(raw_batch)

        get_batch_size = get_batch_size_in_tokens \
            if batch_limit_by_tokens else \
            get_batch_size_in_samples

        current_batch = []

        def get_key(s):
            key = 0
            for e in s:
                key = key * 512 + len(e)
            return key

        if sort_by_length:
            prepared_samples = sorted(
                self.samples, key=lambda s: get_key(s), reverse=True
            )
        elif shuffle:
            __state = random.getstate()
            if self.__random_state is None:
                self.__random_state = random.getstate()
            random.setstate(self.__random_state)
            prepared_samples = [e for e in self.samples]
            random.shuffle(prepared_samples)
            self.__random_state = random.getstate()
            random.setstate(__state)
        else:
            prepared_samples = self.samples

        for sample in prepared_samples:
            current_batch.append(sample)
            if len(current_batch
                  ) > 1 and get_batch_size(current_batch) > batch_size_limit:
                yield get_torch_batch(current_batch[:-1])
                current_batch = current_batch[-1:]
        if len(current_batch[0]) > 0:
            yield get_torch_batch(current_batch)


@configured('data')
def get_train_dataset(
    train_root_path: str = './data/train',
    src_lang_code: str = 'en',
    tgt_lang_code: str = 'fa',
    src_lowercase: bool = False,
    tgt_lowercase: bool = False,
    src_normalizer: str = 'default',
    tgt_normalizer: str = 'default',
    src_tokenizer: str = 'default',
    tgt_tokenizer: str = 'default',
    shared_vocabulary: bool = False
):
    logger = get_logger()

    if os.path.isfile('{}.dataset'.format(train_root_path)):
        logger.info(
            'Train dataset has been already prepared. Loading from binary form ...'
        )
        corpora = Corpora.load('{}.dataset'.format(train_root_path))
        logger.info('Done.')
        return corpora

    field_specs = [
        [src_lowercase, src_normalizer, src_tokenizer],
        [tgt_lowercase, tgt_normalizer, tgt_tokenizer]
    ]

    vocabulary = None
    if shared_vocabulary:
        vocabulary = Vocabulary()

    fields = [
        Field(l, vocabulary, n, t, force_vocabulary_update=True)
        for l, n, t in field_specs
    ]

    corpora = Corpora(fields)

    logger.info('Preparing train corpora ...')
    with open('{}.{}'.format(train_root_path, src_lang_code)) as src_stream, \
            open('{}.{}'.format(train_root_path, tgt_lang_code)) as tgt_stream:
        for src_sentence, tgt_sentence in zip(src_stream, tgt_stream):
            if src_sentence.strip() and tgt_sentence.strip():
                corpora.append([src_sentence, tgt_sentence])
    logger.info('Saving dataset ...')
    corpora.save('{}.dataset'.format(train_root_path))
    logger.info('Saving vocabular(y|ies) ...')
    if shared_vocabulary:
        vocabulary.save_as_csv(
            '{}/vocab.shared'.format(os.path.dirname(train_root_path))
        )
    else:
        fields[0].vocabulary.save_as_csv(
            '{}/vocab.{}'.format(
                os.path.dirname(train_root_path), src_lang_code
            )
        )
        fields[1].vocabulary.save_as_csv(
            '{}/vocab.{}'.format(
                os.path.dirname(train_root_path), tgt_lang_code
            )
        )
    logger.info('Done.')

    return corpora


@configured('data')
def get_validation_dataset(
    train_root_path: str,
    src_lang_code: str,
    tgt_lang_code: str,
    src_lowercase: bool,
    tgt_lowercase: bool,
    src_normalizer: str,
    tgt_normalizer: str,
    src_tokenizer: str,
    tgt_tokenizer: str,
    shared_vocabulary: bool,
    validation_root_path: str = './data/validation'
):
    logger = get_logger()

    if os.path.isfile('{}.dataset'.format(validation_root_path)):
        logger.info(
            'Validation dataset has been already prepared. Loading from binary form ...'
        )
        corpora = Corpora.load('{}.dataset'.format(validation_root_path))
        logger.info('Done.')
        return corpora

    field_specs = [
        [src_lowercase, src_normalizer, src_tokenizer],
        [tgt_lowercase, tgt_normalizer, tgt_tokenizer]
    ]

    vocabularies = None
    if shared_vocabulary:
        vocabulary = Vocabulary()
        vocabulary.load_from_csv(
            '{}/vocab.shared'.format(os.path.dirname(train_root_path))
        )
        vocabularies = [vocabulary, vocabulary]
    else:
        vocabularies = [
            Vocabulary().load_from_csv(
                '{}/vocab.{}'.format(
                    os.path.dirname(train_root_path), src_lang_code
                )
            ),
            Vocabulary().load_from_csv(
                '{}/vocab.{}'.format(
                    os.path.dirname(train_root_path), tgt_lang_code
                )
            )
        ]

    fields = [
        Field(l, v, n, t) for (l, n, t), v in zip(field_specs, vocabularies)
    ]

    corpora = Corpora(fields)

    logger.info('Preparing validation corpora ...')
    with open('{}.{}'.format(validation_root_path, src_lang_code)) as src_stream, \
            open('{}.{}'.format(validation_root_path, tgt_lang_code)) as tgt_stream:
        for src_sentence, tgt_sentence in zip(src_stream, tgt_stream):
            if src_sentence.strip() and tgt_sentence.strip():
                corpora.append([src_sentence, tgt_sentence])
    logger.info('Saving dataset ...')
    corpora.save('{}.dataset'.format(validation_root_path))
    logger.info('Done.')

    return corpora


@configured('data')
def get_test_dataset(
    train_root_path: str,
    src_lang_code: str,
    tgt_lang_code: str,
    src_lowercase: bool,
    tgt_lowercase: bool,
    src_normalizer: str,
    tgt_normalizer: str,
    src_tokenizer: str,
    tgt_tokenizer: str,
    shared_vocabulary: bool,
    test_root_path: str = './data/test'
):
    logger = get_logger()

    if os.path.isfile('{}.dataset'.format(test_root_path)):
        logger.info(
            'Validation dataset has been already prepared. Loading from binary form ...'
        )
        corpora = Corpora.load('{}.dataset'.format(test_root_path))
        logger.info('Done.')
        return corpora

    field_specs = [
        [src_lowercase, src_normalizer, src_tokenizer],
        [tgt_lowercase, tgt_normalizer, tgt_tokenizer]
    ]

    vocabularies = None
    if shared_vocabulary:
        vocabulary = Vocabulary()
        vocabulary.load_from_csv(
            '{}/vocab.shared'.format(os.path.dirname(train_root_path))
        )
        vocabularies = [vocabulary, vocabulary]
    else:
        vocabularies = [
            Vocabulary().load_from_csv(
                '{}/vocab.{}'.format(
                    os.path.dirname(train_root_path), src_lang_code
                )
            ),
            Vocabulary().load_from_csv(
                '{}/vocab.{}'.format(
                    os.path.dirname(train_root_path), tgt_lang_code
                )
            )
        ]

    fields = [
        Field(l, v, n, t) for (l, n, t), v in zip(field_specs, vocabularies)
    ]

    corpora = Corpora(fields)

    logger.info('Preparing validation corpora ...')
    with open('{}.{}'.format(test_root_path, src_lang_code)) as src_stream, \
            open('{}.{}'.format(test_root_path, tgt_lang_code)) as tgt_stream:
        for src_sentence, tgt_sentence in zip(src_stream, tgt_stream):
            if src_sentence.strip() and tgt_sentence.strip():
                corpora.append([src_sentence, tgt_sentence])
    logger.info('Saving dataset ...')
    corpora.save('{}.dataset'.format(test_root_path))
    logger.info('Done.')

    return corpora


def prepare_data():
    get_train_dataset()
    get_validation_dataset()
    get_test_dataset()
