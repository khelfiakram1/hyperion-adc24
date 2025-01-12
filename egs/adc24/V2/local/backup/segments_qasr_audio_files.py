#!/usr/bin/env python


import logging
import os
import sys
import time
import librosa

import numpy as np
from jsonargparse import ArgumentParser
from hyperion.hyp_defs import config_logger
from hyperion.io import AudioWriter as Writer
from hyperion.io import RandomAccessAudioReader as AR




def segment_audio(segments_file, audio_path, output_dir, output_scp,  **kwargs):
    input_args = AR.filter_args(**kwargs)
    output_args = Writer.filter_args(**kwargs)
    logging.info(f"input_args={input_args}")
    logging.info(f"output_args={output_args}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    count = 0
    t1 = time.time()
    wav_scp_entries =[]


    with AR(recordings_file, **input_args) as reader, Writer(
        output_path, **output_args
    ) as writer:
        with open(segments_file, 'r') as seg_file:
            for line in seg_file:
                parts = line.strip().split()
                if len(parts) != 3:
                    logging.warning(f"Invalid line format: {line}")
                    continue

                seg_id, start_time, end_time = parts[0], float(parts[1]), float(parts[2])
                file_path = os.path.join(recordings_dir, f"{seg_id}.wav")
                # Read the corresponding audio data by ID
                try:
                    audio, fs = reader.read([file_path])
                    audio = audio[0]  # Extract the audio array for simplicity
                    if fs != target_fs:
                            audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
                            fs = target_fs
                    start_sample = int(start_time * fs)
                    end_sample = int(end_time * fs)

                    # Cut the segment and write to output
                    segment = audio[start_sample:end_sample]
                    if len(segment) > 0:
                        output_filename = os.path.join(output_path, f"{seg_id}_seg.wav")
                        logging.info(f"Writing segment {seg_id} from {start_time} to {end_time} seconds")
                        writer.write([output_filename], [segment], [fs])
                        count += 1
                    else:
                        logging.warning(f"Empty segment for {seg_id} from {start_time} to {end_time}")

                except KeyError:
                    logging.error(f"ID {seg_id} not found in recordings")

    logging.info(f"Finished generating {count} segments, elapsed-time={time.time() - t1:.2f} seconds")


def main():
    parser = ArgumentParser(description="Generates segmented audio files based on a segments file")

    parser.add_argument("--recordings-file", required=True, help="Input recordings file (e.g., wav.scp)")
    parser.add_argument("--segments-file", required=True, help="Text file with ID, StartTime, EndTime")
    parser.add_argument("--output-path", required=True, help="Directory to save segmented audio files")
    parser.add_argument("--target-fs", required=True, help="Fs")
    parser.add_argument("--format", required=True, help="Fs")
    AR.add_class_args(parser)
    Writer.add_class_args(parser)

    parser.add_argument("-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int, help="Verbose level")
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    segment_audio(args.recordings_file, args.segments_file, args.output_path, **vars(args))

if __name__ == "__main__":
    main()