#!/usr/bin/env python

import os
import soundfile as sf

from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo

import logging
import os
import sys
import time
import librosa

import numpy as np

from jsonargparse import ArgumentParser
from hyperion.hyp_defs import config_logger

import numpy as np
from jsonargparse import ArgumentParser
from hyperion.io import AudioWriter as Writer
from hyperion.io import RandomAccessAudioReader as AR


def segment_audio(segments_file, audio_path, output_dir, output_scp, samplerate, verbose, **kwargs):
    config_logger(verbose)
    logging.info("starting processing")
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    with open(output_scp, 'w') as wav_scp:
        with open(segments_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    logging.info("Skippin malformed line %s", line)
                    continue

                segment_id, start_time, end_time = parts[:3]
                start_time = float(start_time)
                end_time = float(end_time)
                duration = end_time - start_time

                start_time_formatted = f"{int(start_time * 1000):06d}"
                end_time_formatted = f"{int(end_time * 1000):06d}"

                wav_path = os.path.join(audio_path, f"{segment_id}.wav")

                try:
                    data, samplerate = sf.read(wav_path, start=int(start_time * samplerate), stop=int(end_time * samplerate))
                except Exception as e:
                    logging.info("Error reading audio file %s", wav_path)
                    continue

                output_filename = f"{segment_id}_{start_time_formatted}-{end_time_formatted}.wav"
                output_path = os.path.join(output_dir, output_filename)

                try:
                    sf.write(output_path, data, samplerate, format='WAV')
                    logging.info("Segment %s written to %s", segment_id, output_path)
                    count += 1
                except Exception as e:
                    logging.info("Error writing %s : %s", output_path, e)
                
                    continue

                wav_scp.write(f"{segment_id}_{start_time_formatted}-{end_time_formatted} {os.path.abspath(output_path)}\n")
                logging.info("successfully prepared %d segments", count)


def main(): 
    parser = ArgumentParser()
    parser.add_argument("--segments-file", type=str, required=True)
    parser.add_argument("--audio-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--output-scp", type=str, required=True)
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)
    args = parser.parse_args()
   
    segment_audio(**namespace_to_dict(args))


if __name__ == "__main__":
    main()

                




