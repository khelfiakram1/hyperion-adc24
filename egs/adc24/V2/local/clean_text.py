#!/usr/bin/env python3
# Copyright       2025  Ecole de Technologie Superieure (authors: Khelfi Mohammed Akram)

# Script for cleaning Arabic Text 

import re 
import logging 
from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo
from hyperion.hyp_defs import config_logger


def clean_text(text):
    """
    - Remove all non Arabic characters
    - Normalizes Arabic characters
    - Remove diacritics
    - Remove extra spaces
    """
    
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = arabic_diacritics.sub('', text)

    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ى', 'ي')
    text = text.replace('ة', 'ه')
    text = text.replace('ء', '')
    text = re.sub(r'ـ', '', text)

    punctuation_pattern = r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟]'
    text = re.sub(punctuation_pattern, '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_text_file(source_file, target_file,text_only, verbose):
    config_logger(verbose)
    logging.info(f" Cleaning text file")
    with open(source_file, "r", encoding="utf-8") as f, \
        open(target_file, "w", encoding="utf-8") as o:
        for line in f: 
            
            line = line.strip()

            if text_only:
                cleaned_text = clean_text(line)
                o.write(f"{cleaned_text}\n")
            else : 
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    logging.info(f"Cleaning text for utterance {utt_id}")
                    cleaned_text = clean_text(text)
                    o.write(f"{utt_id} {cleaned_text}\n")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--source-file", type=str, required=True, help="Source text file")
    parser.add_argument("--target-file", type=str, required=True, help="Target text file")
    parser.add_argument("--text-only", action="store_true", 
                        help="If set, the script will treat each line as text without utt_id.")    
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)
    args = parser.parse_args()
    config_logger(args.verbose)
    clean_text_file(**namespace_to_dict(args))