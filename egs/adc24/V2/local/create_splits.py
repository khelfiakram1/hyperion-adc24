#!/usr/bin/env python3

import os
import sys
from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse.cut import Cut


from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo

import logging
from hyperion.hyp_defs import config_logger
import random

def split_dataset(recordings_path, supervisions_path, output_dir, verbose):
    config_logger(verbose)
    random.seed(42)
    
    os.makedirs(output_dir, exist_ok=True)

    recordings = RecordingSet.from_jsonl(recordings_path)
    supervisions = SupervisionSet.from_jsonl(supervisions_path)
    all_ids = list(recordings.ids)


    random.shuffle(all_ids)

    total = len(all_ids)
    train_size = int(total * 0.7)
    dev_size = int(total * 0.1)

    train_ids = set(all_ids[:train_size])
    dev_ids = set(all_ids[train_size:train_size + dev_size])
    test_ids = set(all_ids[train_size + dev_size:])
    
    train_recordings = recordings.filter(lambda r: r.id in train_ids)
    dev_recordings = recordings.filter(lambda r: r.id in dev_ids)
    test_recordings = recordings.filter(lambda r: r.id in test_ids)

    train_supervisions = supervisions.filter(lambda s: s.recording_id in train_ids)
    dev_supervisions = supervisions.filter(lambda s: s.recording_id in dev_ids)
    test_supervisions = supervisions.filter(lambda s: s.recording_id in test_ids)


    splits = {
        "train": (train_recordings, train_supervisions),
        "dev": (dev_recordings, dev_supervisions),
        "test": (test_recordings, test_supervisions),
    }



    os.makedirs(output_dir, exist_ok=True)

    for split_name, (split_recordings, split_supervisions) in splits.items():
        
        recordings_output_path = os.path.join(output_dir, f'recordings_{split_name}.jsonl.gz')
        supervisions_output_path = os.path.join(output_dir, f'supervisions_{split_name}.jsonl.gz')
    
        split_recordings.to_jsonl(recordings_output_path)
        split_supervisions.to_jsonl(supervisions_output_path)

        
    logging.info(f"Created splits in {output_dir}")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--recordings-path", type=str, required=True, help="Path to recordings.jsonl.gz")
    parser.add_argument("--supervisions-path", type=str, required=True, help="Path to supervisions.jsonl.gz")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    split_dataset(**namespace_to_dict(args))