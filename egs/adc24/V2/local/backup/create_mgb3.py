import os
import numpy as np
from scipy.io.wavfile import write
from collections import defaultdict
import pandas as pd
from datasets import load_dataset,Audio

ds = load_dataset("MightyStudent/Egyptian-ASR-MGB-3")



def group_and_save_audio(dataset, output_dir):
    """
    Group and save audio files from a dataset by combining parts into a single .wav file.
    
    Args:
        dataset (list of dict): List of dictionaries containing 'audio' and 'sentence'.
        output_dir (str): Directory where the grouped .wav files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    
    grouped_audio = defaultdict(list)
    sampling_rate = None

    # Group audio arrays by prefix
    for entry in dataset['train']:
        audio = entry['audio']
        file_name = audio['path']
        audio_array = np.array(audio['array'])
        sampling_rate = audio['sampling_rate']  # Assuming all files have the same sampling rate

        # Extract the group prefix (everything before `_part_`)
        group_prefix = "_".join(file_name.split("_")[:-2])
        grouped_audio[group_prefix].append(audio_array)
    
    # Save grouped audio files
    for group, audio_parts in grouped_audio.items():
        # Concatenate all audio parts for the group
        combined_audio = np.concatenate(audio_parts)
        
        # Output file name
        output_file_name = f"{group}.wav"
        output_path = os.path.join(output_dir, output_file_name)
        
        # Save the combined audio
        write(output_path, sampling_rate, combined_audio)
        print(f"Saved grouped file: {output_path}")



output_directory = "/export/fs05/mkhelfi1/mgb3/mgb3/mgb3/wav"
group_and_save_audio(ds, output_directory)
