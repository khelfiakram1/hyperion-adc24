#!/usr/bin/env python3

import datasets
import os
import sys
import soundfile as sf
from tqdm import tqdm

from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo

import logging
from hyperion.hyp_defs import config_logger



def prepare_data(output_dir,verbose):
    config_logger(verbose)
    dataset_name = "MightyStudent/Egyptian-ASR-MGB-3"
    logging.info(f"Downloading dataset {dataset_name}")
    dataset = datasets.load_dataset(dataset_name)

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    
    counter = 0

    metadata_path = os.path.join(output_dir, "metadata.csv")
    logging.info(f"Writing metadata to {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("path,sentence\n")
        for example in tqdm(dataset["train"],desc="Processing dataset"):
            try:
                audio = example["audio"]
                array = audio["array"]
                audio_filename = audio["path"]
                sampling_rate = audio["sampling_rate"]
                sentence = example["sentence"]

                audio_filepath = os.path.join(clips_dir, audio_filename)
                sf.write(audio_filepath, array, sampling_rate)
                
                sentence_clean = sentence.replace('\n', ' ').replace('\r', '')
                sentence_clean = sentence_clean.replace(',', '')
                sentence_clean = sentence_clean.replace('...', '') 
                sentence_clean = ' '.join(sentence_clean.split()) 

                f.write(f"{audio_filename},{sentence_clean}\n")
                counter += 1
            
            except Exception as e:
                logging.error(f"Error processing example {example}: {e}")
                continue

    logging.info(f"Processed {counter} examples for mgb3 data")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    prepare_data(**namespace_to_dict(args))





            
    

