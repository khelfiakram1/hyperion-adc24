#!/usr/bin/env python3

import os
import sys
from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse.cut import Cut


from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo

import logging
from hyperion.hyp_defs import config_logger


def split_dataset(recordings_path, supervisions_path, output_dir, seed, verbose):
    config_logger(verbose)
    proportions = [0.8, 0.1, 0.1]

    recordings = RecordingSet.from_file(recordings_path)
    supervisions = SupervisionSet.from_file(supervisions_path)

    cuts = CutSet.from_manifests(recordings, supervisions)
    shuffled_cuts = cuts.shuffle()

    assert abs(sum(proportions) - 1.0) < 1e-6, "Proportions must sum to 1."

    total_cuts = len(cuts)
    train_end = int(total_cuts * proportions[0])
    dev_end = train_end + int(total_cuts * proportions[1])

    train_cuts = shuffled_cuts[:train_end]
    dev_cuts = shuffled_cuts[train_end:dev_end]
    test_cuts = shuffled_cuts[dev_end:]


    os.makedirs(output_dir, exist_ok=True)
    train_supervisions = supervisions.filter(lambda s: s.recording_id in set(cut.recording_id for cut in train_cuts))
    dev_supervisions = supervisions.filter(lambda s: s.recording_id in set(cut.recording_id for cut in dev_cuts))
    test_supervisions = supervisions.filter(lambda s: s.recording_id in set(cut.recording_id for cut in test_cuts))
    
    splits = {
        'train': (train_cuts.recordings(), train_supervisions),
        'validation': (dev_cuts.recordings(), dev_supervisions),
        'test': (test_cuts.recordings(), test_supervisions)
    }

    for split_name, (split_recordings, split_supervisions) in splits.items():
        
        recordings_output_path = os.path.join(output_dir, f'recordings_{split_name}.jsonl.gz')
        supervisions_output_path = os.path.join(output_dir, f'supervisions_{split_name}.jsonl.gz')
        
        split_recordings.to_file(recordings_output_path)
        split_supervisions.to_file(supervisions_output_path)


        # split_recordings = split_cuts.recordings()
        # split_supervisions = split_cuts.supervisions()

        # split_recodings_path = os.path.join(output_dir, f'recordings_{split_name}.jsonl.gz')
        # split_supervisions_path = os.path.join(output_dir, f'supervisions_{split_name}.jsonl.gz')

        # split_recordings.to_file(split_recodings_path)
        # split_supervisions.to_file(split_supervisions_path)


    logging.info("Split dataset done ")
       

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--recordings-path", type=str, required=True, help="Path to recordings.jsonl.gz")
    parser.add_argument("--supervisions-path", type=str, required=True, help="Path to supervisions.jsonl.gz")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    split_dataset(**namespace_to_dict(args))