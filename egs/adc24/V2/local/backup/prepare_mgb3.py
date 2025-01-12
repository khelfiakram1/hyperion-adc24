#!/usr/bin/env python

import sys
print(sys.version)
print('hello')
from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo

import logging
import os
import glob
import sys
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

from hyperion.hyp_defs import config_logger

def process_directory(input_dir,wav_dir, output_segments_dir, output_metadata_dir):

    processed_files = 0
    logging.info("Preparing corpus %s -> %s", input_dir, output_segments_dir)
    os.makedirs(output_segments_dir, exist_ok=True)
    os.makedirs(output_metadata_dir, exist_ok=True)

    wav_scp_path = os.path.join(output_metadata_dir, "wav.scp")
    utt2dur_path = os.path.join(output_metadata_dir, "utt2dur")
    utt2seq_path = os.path.join(output_metadata_dir, "utt2seq")
    utt2bw_path = os.path.join(output_metadata_dir, "utt2bw")

    with open(os.path.join(input_dir,"Ali", "segments"), "r") as segments_file, \
         open(os.path.join(input_dir,"Ali", "text_noverlap"), "r") as text_noverlap_file, \
         open(os.path.join(input_dir,"Ali", "text_noverlap.bw"), "r") as text_noverlap_bw_file, \
         open(wav_scp_path, "w", encoding="utf-8") as wav_scp, \
         open(utt2dur_path, "w", encoding="utf-8") as utt2dur, \
         open(utt2seq_path, "w", encoding="utf-8") as utt2seq, \
         open(utt2bw_path, "w", encoding="utf-8") as utt2bw:
        
        text_mapping = {line.split()[0]: " ".join(line.split()[1:]) for line in text_noverlap_file}
        text_bw_mapping = {line.split()[0]: " ".join(line.split()[1:]) for line in text_noverlap_bw_file}

        for line in tqdm(segments_file):
            segment_id, main_file, start, end = line.strip().split()
            start, end = float(start), float(end)
            main_file_path = os.path.join(wav_dir, f"{main_file}.wav")
            segment_output_path = os.path.join(output_segments_dir, f"{segment_id}.wav")

            if segment_id in text_mapping and segment_id in text_bw_mapping:
                try:
                    audio,sr = librosa.load(main_file_path,sr=None, offset=start, duration=end-start)
                    sf.write(segment_output_path, audio, sr)
                    wav_scp.write(f"{segment_id} {segment_output_path}\n")
                    duration = librosa.get_duration(y=audio, sr=sr)
                    utt2dur.write(f"{segment_id} {duration:.2f}\n")
                    utt2seq.write(f"{segment_id} {text_mapping[segment_id]}\n")
                    utt2bw.write(f"{segment_id} {text_bw_mapping[segment_id]}\n")


                    processed_files += 1


                except Exception as e:
                    logging.error(f"Error processing {segment_id}: {e}")
                    continue  
            else:
                logging.warning(f"Skipping {segment_id}: No transcription found")

def prepare_mgb3(corpus_dir, output_segments_dir, output_metadata_dir, verbose):
    config_logger(verbose)
    for sub_dir in ["adapt","test","dev"]:
        input_dir = os.path.join(corpus_dir, sub_dir)
        wav_dir = os.path.join(corpus_dir, "wav")
        output_segments_subdir = os.path.join(output_segments_dir, sub_dir)
        output_metadata_subdir = os.path.join(output_metadata_dir, sub_dir)

        process_directory(input_dir, wav_dir, output_segments_subdir, output_metadata_subdir)


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepare MGB3 dataset for segmentation and transcription")
    parser.add_argument("--corpus-dir", required=True, help="Path to the directory of MGB3 dataset")
    parser.add_argument("--output-segments-dir", required=True, help="Path to the output directory for the segmented audio files")
    parser.add_argument("--output-metadata-dir", required=True, help="Path to the output directory for the metadata files")
    parser.add_argument("-v", "--verbose", default=1, choices=[0, 1, 2, 3], type=int, help="Verbosity level")
    args = parser.parse_args()

    prepare_mgb3(**namespace_to_dict(args))

