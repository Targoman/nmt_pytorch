# coding: utf-8
"""
Routines related to training a neural machine translation model
"""

import math
import os
import random
import time

import numpy as np
# pylint: disable=no-member
import torch
from torch import nn
from torch.optim import Optimizer

from nmt.common import Ignore, configured, get_logger, get_device, mark_optimization_step, set_random_seeds
from nmt.dataset import get_train_dataset, get_validation_dataset
from nmt.encoderdecoder import EncoderDecoder
from nmt.loss import get_loss_function
from nmt.model import build_model
from nmt.optimization import build_optimizer, build_scheduler
from nmt.predict import evaluate

# pylint: disable=no-value-for-parameter

@configured('train')
def initialize(
    model: nn.Module,
    embed_init: str = 'normal',
    embed_init_gain: float = 1.0,
    embed_init_scale: float = 0.01,
    weight_init: str = 'xavier',
    weight_init_gain: float = 1.0,
    weight_init_scale: float = 0.01,
    bias_init: str = 'xavier',
    bias_init_gain: float = 1.0,
    bias_init_scale: float = 0.01
):
    def __create_initializer(method, scale, gain):
        if method == 'xavier':
            return lambda p: nn.init.xavier_normal_(p, gain)
        elif method == 'normal':
            return lambda p: nn.init.normal_(p, std=scale)
        else:
            raise RuntimeError('Invalid initializer `{}`.'.format(method))

    init_embed_weights = __create_initializer(
        embed_init, embed_init_scale, embed_init_gain
    )
    init_weights = __create_initializer(
        weight_init, weight_init_scale, weight_init_gain
    )
    init_bias = __create_initializer(bias_init, bias_init_scale, bias_init_gain)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if "embed" in name:
                if p.dim() > 1:
                    init_embed_weights(p.data)
                else:
                    nn.init.zeros_(p.data)
                if "src" in name:
                    p.data[model.src_vocab.pad_index].zero_()
                elif "tgt" in name:
                    p.data[model.tgt_vocab.pad_index].zero_()
            elif "bias" in name:
                if p.dim() > 1:
                    init_bias(p.data)
                else:
                    nn.init.zeros_(p.data)
            elif "f_params" in name:
                pass
            else:
                if p.dim() > 1:
                    init_weights(p.data)
                else:
                    nn.init.ones_(p.data)

@configured('train')
def train(
    max_steps: int = 100,
    batch_size_limit: int = 400,
    batch_limit_by_tokens: bool = True,
    report_interval_steps: int = 10,
    validation_interval_steps: int = 100,
    lr_scheduler_at: str = 'every_step',
    n_ckpts_to_keep: int = 3,
    teacher_forcing: bool = True,
    random_seed: int = 42
):

    set_random_seeds(random_seed)
    logger = get_logger()

    train_dataset = get_train_dataset()
    assert len(
        train_dataset.fields
    ) >= 2, "Train dataset must have at least two fields (source and target)."
    validation_dataset = get_validation_dataset()
    assert len(
        validation_dataset.fields
    ) >= 2, "Validation dataset must have at least two fields (source and target)."

    loss_function = get_loss_function(
        train_dataset.fields[1].vocabulary.pad_index
    )

    model = build_model(
        train_dataset.fields[0].vocabulary,
        train_dataset.fields[1].vocabulary
    )
    
    model.to(get_device())
    loss_function.to(get_device())

    optimizer = build_optimizer(model.parameters())
    scheduler = build_scheduler(optimizer)

    initialize(model)

    def noop():
        return None

    def step_lr_scheduler():
        return scheduler.step()

    run_scheduler_at_step = noop
    run_scheduler_at_validation = noop
    run_scheduler_at_epoch = noop

    if scheduler is not None:
        if lr_scheduler_at == 'every_step':
            run_scheduler_at_step = step_lr_scheduler
        elif lr_scheduler_at == 'every_validation':
            run_scheduler_at_validation = step_lr_scheduler
        elif lr_scheduler_at == 'every_epoch':
            run_scheduler_at_epoch = step_lr_scheduler

    step = 0
    epoch = 0

    kept_checkpoint_path_score_map = {}

    best_checkpoint_specs = {"score": -math.inf, "step": -1}

    @configured('model')
    def maybe_save_checkpoint(score: Ignore[float], output_path: str):

        if len(kept_checkpoint_path_score_map) < n_ckpts_to_keep or \
                any(score > s for s in kept_checkpoint_path_score_map.values()):
            if len(kept_checkpoint_path_score_map) >= n_ckpts_to_keep:
                worst_checkpoint_path = sorted(
                    kept_checkpoint_path_score_map.keys(),
                    key=lambda p: kept_checkpoint_path_score_map[p],
                    reverse=False
                )
                worst_checkpoint_path = worst_checkpoint_path[0]
                kept_checkpoint_path_score_map.pop(worst_checkpoint_path)
                try:
                    os.unlink(worst_checkpoint_path)
                except:
                    logger.warn(
                        'Could not unlink {}.'.format(worst_checkpoint_path)
                    )

            if score > best_checkpoint_specs["score"]:
                logger.info(
                    'New `best model` found with score {:.3f} at step {}.'.
                    format(score, step)
                )
                best_checkpoint_specs["score"] = score
                best_checkpoint_specs["step"] = step

            state_dict = {
                "step":
                    step,
                "best_checkpoint_specs":
                    best_checkpoint_specs,
                "model_state":
                    model.state_dict(),
                "optimizer_state":
                    optimizer.state_dict(),
                "scheduler_state":
                    scheduler.state_dict() if scheduler is not None else None
            }
            checkpoint_path = '{}/step_{}_score_{:.3f}.pt'.format(
                output_path, step, score
            )
            torch.save(state_dict, checkpoint_path)
            kept_checkpoint_path_score_map[checkpoint_path] = score

    model.train()

    validation_done_already = False
    while step < max_steps:

        start_time = time.time()
        total_tokens_processed = 0
        for batch in train_dataset.iterate(
            get_device(),
            batch_size_limit,
            batch_limit_by_tokens
        ):
            step += 1
            if step >= max_steps:
                break

            x_mask = batch[0] != model.src_vocab.pad_index
            x_mask = x_mask.unsqueeze(1)

            y_mask = batch[1] != model.tgt_vocab.pad_index
            y_mask = y_mask.unsqueeze(1)

            x_e = model.encode(batch[0], x_mask)
            log_probs = model.decode(
                batch[1][:, :-1],
                x_e,
                y_mask[:, :, :-1],
                x_mask,
                teacher_forcing=teacher_forcing
            )
            token_count = y_mask[:, :, 1:].sum().item()
            loss = loss_function(log_probs, batch[1][:, 1:], model.get_target_embeddings()) / token_count
            loss.backward()

            optimizer.step()
            mark_optimization_step()
            optimizer.zero_grad()

            run_scheduler_at_step()

            total_tokens_processed += token_count

            if step > 0 and step % report_interval_steps == 0:
                elapsed_time = time.time() - start_time
                baseline_loss = loss_function.uniform_baseline_loss(
                    log_probs, batch[1][:, 1:]
                )
                logger.info(
                    'Epoch_{} Step_{}: loss={:.3f}(vs {:.3f} uniform), tokens/s={:.1f}, lr={}'
                    .format(
                        epoch, step, loss.item(), baseline_loss,
                        total_tokens_processed / elapsed_time,
                        optimizer.param_groups[0]['lr']
                    )
                )
                start_time = time.time()
                total_tokens_processed = 0

            if step > 0 and step % validation_interval_steps == 0:
                log_prefix = 'Epoch_{} Step_{}'.format(epoch, step)
                score = evaluate(
                    validation_dataset, log_prefix, model, loss_function
                )
                maybe_save_checkpoint(score)
                model.train()
                run_scheduler_at_validation()
                start_time = time.time()
                total_tokens_processed = 0
                validation_done_already = True
            else:
                validation_done_already = False

        epoch += 1
        logger.info('Epoch {} finished.'.format(epoch))
        run_scheduler_at_epoch()

    if not validation_done_already:
        log_prefix = 'Final (epoch={} ~ step={})'.format(epoch, step)
        score = evaluate(validation_dataset, log_prefix, model, loss_function)
        maybe_save_checkpoint(score)
    logger.info(
        'Best validation loss was {:.3f} at step {}.'.format(
            best_checkpoint_specs["score"], best_checkpoint_specs["step"]
        )
    )
