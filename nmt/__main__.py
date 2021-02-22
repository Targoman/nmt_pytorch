# coding: utf-8
"""
Entrypoint of command line utility
"""

import os
import json
import argparse

from nmt.common import get_configuration, set_configuration_file_path, make_logger, logger
from nmt.dataset import prepare_data, get_validation_dataset, get_test_dataset
from nmt.train import train
from nmt.predict import predict, evaluate, get_vocabularies
from nmt.sanity import sanity_check
from nmt.model import update_and_ensure_model_output_path

def run_evaluate(get_dataset, log_prefix):
    dataset = get_dataset()
    evaluate(dataset, log_prefix)

def main():
    parser = argparse.ArgumentParser('nmt')
    parser.add_argument(
        '-m',
        '--mode',
        choices=[
            'save_config', 'prepare_data', 'train', 'test', 'translate',
            'sanity_check'
        ],
        help='Mode of execution.',
        required=True
    )
    parser.add_argument(
        '-x',
        '--index',
        type=int,
        help='Model index to be used in case of "test/translate" modes.',
        required=False
    )
    parser.add_argument(
        '-c',
        '--config_path',
        type=str,
        help='Path to the JSON formatted config file.',
        required=True
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='Path to input text file.',
        required=False
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Path to output hypotheses file.',
        required=False
    )

    args = parser.parse_args()

    if args.mode != 'save_config':
        with open(args.config_path) as f:
            get_configuration().load(json.load(f))
        set_configuration_file_path(os.path.dirname(args.config_path))
        update_and_ensure_model_output_path(args.mode, args.index)
        make_logger()

    if args.mode == 'save_config':
        with open(args.config_path, 'w') as f:
            json.dump(get_configuration().get_as_dict(), f, indent=4)
    elif args.mode == 'prepare_data':
        prepare_data()
    elif args.mode == 'train':
        train()
    elif args.mode == 'test':
        run_evaluate(get_validation_dataset, 'validation')
        run_evaluate(get_test_dataset, 'test')
    elif args.mode == 'translate':
        if not args.input or not args.output:
            raise ValueError('In order to use translate command, both input and output path must be specified.')
        predict(args.input, args.output, 'translate')
    elif args.mode == 'sanity_check':
        sanity_check()
    else:
        raise ValueError('Invalid execution mode.')

if __name__ == "__main__":
    main()