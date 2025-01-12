#!/usr/bin/env python3

# create manifest for mgb3 data since it don't have lhost for my dataset 

import os
import sys
import pandas as pd
from lhotse import RecordingSet, SupervisionSet, Recording, SupervisionSegment

from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo

import logging
from hyperion.hyp_defs import config_logger


def create_manifest(src_dir,dst_dir,verbose):
    config_logger(verbose)
    metadata_csv = os.path.join(src_dir, "metadata.csv")
    clips_dir = os.path.join(src_dir, "clips")

    try:
        df = pd.read_csv(metadata_csv)
    except Exception as e: 
        logging.info(f"Error reading metadata.csv: {e}")
        sys.exit(1)

    recordings = []
    supervisions = []

    logging.info("Creating recordings and supervisions")

    for idx, row in df.iterrows():
        recording_id = os.path.splitext(row["path"])[0]
        audio_path = os.path.join(clips_dir, row["path"])
        transcript = row["sentence"]

        try:
            recording = Recording.from_file(audio_path, recording_id = recording_id)
            recordings.append(recording)
        
        except Exception as e:
            logging.error(f"Error reading audio file {audio_path}: {e}")
            continue

        segment = SupervisionSegment(
            id=recording_id,
            recording_id=recording_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            text=transcript,
            language='egy',
            speaker='unkown'
        )
        supervisions.append(segment)
    
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    os.makedirs(dst_dir, exist_ok=True)
    recording_set.to_file(os.path.join(dst_dir, "recordings.jsonl.gz"))
    supervision_set.to_file(os.path.join(dst_dir, "supervisions.jsonl.gz"))

    logging.info(f"Manifests created and saved to {dst_dir}")



if __name__ == "__main__":
  
    parser = ArgumentParser()
    parser.add_argument("--src-dir", type=str, required=True, help="source directory")
    parser.add_argument("--dst-dir", type=str, required=True, help="destination directory")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    create_manifest(**namespace_to_dict(args))
 
