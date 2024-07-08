import os

# Paths to your input files
utt2lang_path = '/home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/data/adi17/test_proc_audio_no_sil/utt2lang'
wav_scp_path = '/home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/data/adi17/test_proc_audio_no_sil/wav.scp'

# Paths to your output files
output_utt2lang_combined = '/home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/data/adi17/test_bis/utt2lang'
output_wav_scp_combined = '/home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/data/adi17/test_bis/wav.scp'

# Dialect codes
jordanian_code = 'ara-jor'
palestinian_code = 'ara-pal'

# Function to read and filter data
def filter_and_combine_dialects(input_utt2lang, input_wav_scp, dialect_codes, output_utt2lang, output_wav_scp):
    utt_ids = set()
    
    # Read and filter utt2lang
    with open(input_utt2lang, 'r') as infile, open(output_utt2lang, 'w') as outfile:
        for line in infile:
            utt_id, lang_code = line.strip().split()
            if lang_code in dialect_codes:
                utt_ids.add(utt_id)
                outfile.write(line)
    
    # Read and filter wav.scp
    with open(input_wav_scp, 'r') as infile, open(output_wav_scp, 'w') as outfile:
        for line in infile:
            utt_id, wav_path = line.strip().split(maxsplit=1)
            if utt_id in utt_ids:
                outfile.write(line)

# Filter Jordanian and Palestinian dialects
filter_and_combine_dialects(
    utt2lang_path, wav_scp_path, 
    {jordanian_code, palestinian_code}, 
    output_utt2lang_combined, output_wav_scp_combined
)

print("Data separation and combination complete.")
