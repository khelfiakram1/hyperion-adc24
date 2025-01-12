#!/usr/bin/env python

import os 
import requests

from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo
import logging
from tqdm import tqdm
from hyperion.hyp_defs import config_logger

dialect_to_lang = {
    "MSA": "ara-arb",
    "EGY": "ara-egy",
    "GLF": "ara-glf",
    "LEV": "ara-lev",
    "NOR": "ara-mag"
}

def get_mgb3(wav_list, split_id, files_output, output_dir, verbose, **kwargs):
    config_logger(verbose)
    logging.info("Starting MGB3 data download for %s ", split_id)
    wav_scp_entries = []
    utt2lang_entries = []

    with open(wav_list, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    for url in tqdm(urls):

        parts = url.split('/')
        split = parts[-3]
        dialect = parts[-2]
        filename = parts[-1]
        utt_id = os.path.splitext(filename)[0]

        lang_code = dialect_to_lang.get(dialect)
        if not lang_code :
            logging.info("Skipping %s Language not found", url)
            continue
        
        dest_file_path = os.path.join(output_dir, split, dialect)
        os.makedirs(dest_file_path, exist_ok=True)

        dest_file = os.path.join(dest_file_path, filename)
        if not os.path.exists(dest_file):
            try: 
                response = requests.get(url)
                response.raise_for_status()
                with open(dest_file, 'wb') as f:
                    f.write(response.content)
                logging.info("Downloaded %s", url)
            except requests.exceptions.RequestException as e:
                logging.info("Error downloading %s : %s", url, e)
                continue
        
        abs_path = os.path.abspath(dest_file)
        wav_scp_entries.append(f"{utt_id} {abs_path}")
        utt2lang_entries.append(f"{utt_id} {lang_code}")
    
    wav_scp_path = os.path.join(files_output, split_id, "wav.scp")
    utt2lang_path = os.path.join(files_output, split_id, "utt2lang")

    with open(wav_scp_path, 'w') as f:
        for entry in wav_scp_entries:
            f.write(f"{entry}\n")

    with open(utt2lang_path, 'w') as f:
        for entry in utt2lang_entries:
            f.write(f"{entry}\n")

    
    logging.info("Successfully prepared %d files for %s split", len(wav_scp_entries), split_id)
    
    

    

def main(): 
    parser = ArgumentParser()
    parser.add_argument("--wav-list", type=str, required=True)
    parser.add_argument("--split-id", type=str, required=True)
    parser.add_argument("--files-output", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)
    args = parser.parse_args()
    logging.info("begin")
    get_mgb3(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
