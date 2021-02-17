# coding: utf-8
"""
Routines related to inference on a neural machine translation model
"""

import os
import time
from typing import Callable, Tuple

# pylint: disable=no-member,no-value-for-parameter
import torch

from nmt.common import Ignore, configured, get_logger, get_device

from nmt.dataset import Corpora, Vocabulary, Field
from nmt.model import build_model
from nmt.loss import get_loss_function
from nmt.search import beam_search, short_sent_penalty
from nmt.metric import update_metric_params, Metric, BleuMetric
from nmt.encoderdecoder import EncoderDecoder


@configured('model')
def find_best_model(output_path: str):
    def get_score(path: str):
        return float(path.split('_')[-1][:-3])

    best_model_path = None
    for model_path in (
        model_path for model_path in (
            os.path.join(output_path, model_path)
            for model_path in os.listdir(output_path)
        ) if (
            os.path.isfile(model_path) and
            os.path.splitext(model_path)[1] == '.pt'
        )
    ):
        if best_model_path is None or get_score(best_model_path
                                               ) < get_score(model_path):
            best_model_path = model_path
    return best_model_path

@configured('data')
def get_vocabularies(
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
):
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
        vocabularies = (vocabulary, vocabulary)
    else:
        vocabularies = (
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
        )

    fields = (
        Field(l, v, n, t) for (l, n, t), v in zip(field_specs, vocabularies)
    )
    
    return vocabularies, tuple(fields)

@configured('train')
def predict(
    input: Ignore[str],
    output: Ignore[str],
    log_prefix: Ignore[str],
    model: EncoderDecoder = None,
    batch_size_limit: int = 400,
    batch_limit_by_tokens: bool = True,
):
    
    logger = get_logger()

    (src_vocab, _), (src_field, tgt_field) = get_vocabularies()

    dataset = Corpora([src_field])
    logger.info(f'{log_prefix}: Loading input file ...')
    with open(input) as src_stream:
        for src_sentence in src_stream:
            if src_sentence.strip():
                dataset.append([src_sentence])
    logger.info(f'{log_prefix}: Loading done.')

    if model is None:
        best_model_path = find_best_model()
        if best_model_path is None:
            raise RuntimeError(
                'Model has not been trained yet. Train the model first.'
            )
        model = build_model(
            src_field.vocabulary, tgt_field.vocabulary
        )
        state_dict = torch.load(best_model_path)
        model.load_state_dict(state_dict['model_state'])
        model.to(get_device())

    with open(output, 'w') as output_stream, torch.no_grad():

        for batch in dataset.iterate(
            get_device(),
            batch_size_limit,
            batch_limit_by_tokens,
            sort_by_length=False,
            shuffle=False
        ):
            x_mask = batch[0] != src_vocab.pad_index
            x_mask = x_mask.unsqueeze(1)

            x_e = model.encode(batch[0], x_mask)
            y_hat, _ = beam_search(
                x_e, x_mask, model, get_scores=short_sent_penalty
            )
            sentence = src_field.to_sentence_str(
                batch[0][-1].tolist()
            )
            generated = tgt_field.to_sentence_str(
                y_hat[-1].tolist()
            )

            logger.info('SENTENCE:\n ---- {}'.format(sentence))
            logger.info('GENERATED:\n ---- {}'.format(generated))

            for generated in (src_field.to_sentence_str(s) for s in y_hat.tolist()):
                output_stream.write(f'{generated}\n')


@configured('train')
def evaluate(
    validation_dataset: Corpora,
    log_prefix: Ignore[str],
    model: EncoderDecoder = None,
    loss_function: Callable = None,
    batch_size_limit: int = 400,
    batch_limit_by_tokens: bool = True,
    teacher_forcing: bool = True,
    metrics: Tuple[Metric] = None
):
    assert len(
        validation_dataset.fields
    ) >= 2, "Validation dataset must have at least two fields (source and target)."

    logger = get_logger()

    if loss_function is None:
        loss_function = get_loss_function(
            validation_dataset.fields[1].vocabulary.pad_index
        )
        loss_function.to(get_device())
    if model is None:
        best_model_path = find_best_model()
        if best_model_path is None:
            raise RuntimeError(
                'Model has not been trained yet. Train the model first.'
            )
        model = build_model(
            validation_dataset.fields[0].vocabulary,
            validation_dataset.fields[1].vocabulary
        )
        state_dict = torch.load(best_model_path)
        model.load_state_dict(state_dict['model_state'])
        model.to(get_device())
    pad_index = model.tgt_vocab.pad_index

    total_item_count = 0
    total_validation_loss = 0
    model.eval()

    printed_samples = 0

    if metrics is None:
        metrics = (BleuMetric(),)
    else:
        metrics = (BleuMetric(),) + tuple(m for m in metrics if not isinstance(m, BleuMetric))

    with torch.no_grad():

        start_time = time.time()
        for validation_batch in validation_dataset.iterate(
            get_device(),
            batch_size_limit,
            batch_limit_by_tokens,
            sort_by_length=False,
            shuffle=False
        ):
            x_mask = validation_batch[0] != model.src_vocab.pad_index
            x_mask = x_mask.unsqueeze(1)

            y_mask = validation_batch[1] != model.tgt_vocab.pad_index
            y_mask = y_mask.unsqueeze(1)

            x_e = model.encode(validation_batch[0], x_mask)
            log_probs = model.decode(
                validation_batch[1][:, :-1],
                x_e,
                y_mask[:, :, :-1],
                x_mask,
                teacher_forcing=teacher_forcing
            )

            loss = loss_function(log_probs, validation_batch[1][:, 1:], model.get_target_embeddings())
            total_item_count += y_mask[:, :, 1:].sum().item()
            total_validation_loss += loss.item()

            y_hat, _ = beam_search(
                x_e, x_mask, model, get_scores=short_sent_penalty
            )
            
            if printed_samples < 4:
                sentence = validation_dataset.fields[0].to_sentence_str(
                    validation_batch[0][-1].tolist()
                )
                reference = validation_dataset.fields[1].to_sentence_str(
                    validation_batch[1][-1].tolist()
                )
                generated = validation_dataset.fields[1].to_sentence_str(
                    y_hat[-1].tolist()
                )
                logger.info('SENTENCE:\n ---- {}'.format(sentence))
                logger.info('REFERENCE:\n ---- {}'.format(reference))
                logger.info('GENERATED:\n ---- {}'.format(generated))

                printed_samples += 1

            update_metric_params(y_hat, validation_batch[1], pad_index, metrics)

    elapsed_time = time.time() - start_time
    logger.info(
        f'{log_prefix}: '
        f'evaluation_loss={total_validation_loss / total_item_count:.3f}, '
        f'elapsed_time={int(elapsed_time + 0.5)}s'
    )
    for metric_repr in (str(m) for m in metrics):
        logger.info(f'{log_prefix}: evaluation {metric_repr}')

    return metrics[0].get_score()

