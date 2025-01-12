#!/usr/bin/env python
import sys
print(sys.version)

from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo

import logging
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from hyperion.hyp_defs import config_logger


import xml.etree.ElementTree as ET
from tqdm import tqdm

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def remove_prefix(input_string, prefix):
    if prefix and input_string.startswith(prefix):
        return input_string[len(prefix):]
    return input_string


def parse_segments(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    head = root[0]
    body = root[1]
    segments = body[0]
    speakers_node = head[2]

    speakers_info = [c.attrib for c in speakers_node]

    segment_list = []
    for segment in segments:
        segment_dict = segment.attrib
        words = [e.text for e in segment]
        segment_dict['words'] = words
        segment_dict['utterance'] = ' '.join(words)
        who = segment_dict['who']
        segment_dict['speaker_id'] = remove_prefix(who.split('_')[-2], 'speaker')

        segment_list.append(segment_dict)
        
    
    return segment_list, speakers_info


def prepare_qasr(corpus_dir, output_dir, target_fs, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_dir)
    corpus_dir_path = Path(corpus_dir)


    wav_dir = corpus_dir_path
    xml_dirpath= corpus_dir + '/release/train_20210109/xml'
    logging.info("xml files path %s", xml_dirpath)
    xml_filepaths = glob.glob(f"{xml_dirpath}/*.xml")
    keys = ['starttime', 'endtime', 'speaker_id', 'utterance']
    keys_id = ['starttime', 'endtime']
    sep = ' '

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    seg_output_file = output_dir/ "segments.txt"
    speakers_output_file = output_dir/ "speakers.txt"
    ids_output_file = output_dir/ "ids.txt"

    logging.info("Preparing Transcript files")
    with open(seg_output_file, 'w', encoding='utf-8') as f_seg, \
        open(ids_output_file, 'w', encoding='utf-8') as f_id,\
        open(speakers_output_file, 'w', encoding='utf-8') as f_spe:

        for xml_fpath in tqdm(xml_filepaths):

            segment_list, speakers_info = parse_segments(xml_fpath)

            xml_fname = remove_suffix(os.path.basename(xml_fpath), '.xml')


            for i, segment in enumerate(segment_list):
                segment_str = xml_fname + sep + sep.join(segment[k] for k in keys)
                f_seg.write(segment_str + '\n')     

            for i, segment in enumerate(segment_list):
                segment_str = xml_fname + sep + segment['starttime']+ sep + segment['endtime']
                f_id.write(segment_str + '\n')     
        
            for i, speaker_info in enumerate(speakers_info):
                speaker_str = sep.join(speaker_info.values())
                f_spe.write(speaker_str + '\n')

    logging.info("Prepared text files for QASR")

if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares QASR for training")
    parser.add_argument("--corpus-dir",
                        required=True,
                        help="Path to the original dataset")
    parser.add_argument("--output-dir", required=True, help="data path")

    parser.add_argument("--target-fs",
                        default= 16000,
                        type=int,
                        help="Target sampling frequency")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)
    args = parser.parse_args()

    prepare_qasr(**namespace_to_dict(args))