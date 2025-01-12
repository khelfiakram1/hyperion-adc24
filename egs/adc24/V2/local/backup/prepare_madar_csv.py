#!/usr/bin/env python


import pandas as pd 
import sys
from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo
import logging
from hyperion.hyp_defs import config_logger
import os

def extract_split(split_value):
    parts = split_value.split('-')
    if len(parts) == 3:
        return parts[2]
    else:
        return 'unknown split'


def preprepare_madar(corpus_dir, output_dir):

    egy_file = os.path.join(corpus_dir, f'MADAR.corpus.Cairo.tsv')
    msa_file = os.path.join(corpus_dir, f'MADAR.corpus.MSA.tsv')

    if not os.path.isfile(egy_file):
        logging.error(f"File {egy_file} not found")
        return
    
    if not os.path.isfile(msa_file):
        logging.error(f"File {msa_file} not found")
        return

    egy_df = pd.read_csv(egy_file, sep='\t', encoding='utf-8')    
    msa_df = pd.read_csv(msa_file, sep='\t', encoding='utf-8')

    egy_df['ID'] = egy_df['sentID.BTEC']
    msa_df['ID'] = msa_df['sentID.BTEC']

    egy_df['SPLIT'] = egy_df['split'].apply(extract_split)
    msa_df['SPLIT'] = msa_df['split'].apply(extract_split)

    merged_df = pd.merge(
        egy_df[['ID', 'SPLIT', 'sent']].rename(columns={'sent': 'EGYtranscript'}),
        msa_df[['ID', 'SPLIT', 'sent']].rename(columns={'sent': 'MSAtranscript'}),
        on=['ID', 'SPLIT'],
        how='left'  # Keep all entries from EGY, even if MSA is missing
    )

    splits = merged_df['SPLIT'].unique()
    for split in splits:
        split_df = merged_df[merged_df['SPLIT'] == split]
        out_dir = os.path.join(output_dir, split)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(out_dir, 'segments.csv')
        split_df.to_csv(output_file, index=False, encoding='utf-8')

        logging.info(f"Saved {split} split data to {output_file}")







if __name__ == "__main__":
    parser = ArgumentParser(description="Prepares MADAR data for seq2seq training")

    parser.add_argument("--corpus-dir",
                        required=True,
                        help="Path to the original dataset")


    parser.add_argument("--output-dir",
                        required=True,
                        help="data path")       

    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)

    args = parser.parse_args()
    preprepare_madar(**namespace_to_dict(args))