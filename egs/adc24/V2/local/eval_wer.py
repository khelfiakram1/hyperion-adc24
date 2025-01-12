#!/usr/bin/env python3


from jiwer import wer
from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo

import logging
from hyperion.hyp_defs import config_logger



def eval_wer(source_text, target_text):
    logging.info("Computing WER")
    with open(source_text, 'r', encoding="utf-8") as f:
        y_test = {line.split()[0]:" ".join(line.split()[1:]) for line in f}

    with open(target_text, 'r', encoding="utf-8") as f:
        y_pred = {line.split()[0]:" ".join(line.split()[1:]) for line in f}
    
    common_ids = set(y_test.keys()).intersection(y_pred.keys())
    references = [y_test[id] for id in common_ids]
    hypotheses = [y_pred[id] for id in common_ids]
    logging.info(f"WER: {wer(references, hypotheses)}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source-text', type=str, required=True)
    parser.add_argument('--target-text', type=str, required=True)
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose

    eval_wer(**namespace_to_dict(args))
