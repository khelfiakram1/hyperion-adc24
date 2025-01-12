import os
# Define the mapping from dialects to groups

dialect_groups = {
    "Maghrebi1": ["ara-arq", "ara-mor", "ara-mau","ara-ayl"],
    "EgyptianSudanese": ["ara-arz", "ara-sud","ara-pal", "ara-jor"],
    "Levantine": [ "ara-leb", "ara-syr","ara-yem"],
    "Gulf": ["ara-ksa", "ara-kuw", "ara-oma", "ara-qat", "ara-uae"],
    "Mesopotamian": ["ara-acm"]
    }
# Reverse the mapping for quick lookup: dialect -> group
dialect_to_group = {}
for group, dialects in dialect_groups.items():
    for dialect in dialects:
        dialect_to_group[dialect] = group

# File paths
input_utt2lang = '/home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/data/adi17/train_proc_audio_no_sil/utt2lang'  # Replace with your input file path
output_utt2group = '/home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/data/adi17/train_proc_audio_no_sil/utt2langcopy'  # Replace with your output file path

# Read the original utt2lang file and process each line
with open(input_utt2lang, 'r') as infile, open(output_utt2group, 'w') as outfile:
    for line in infile:
        parts = line.strip().split()
        utt_id = parts[0]  # utterance ID
        dialect = parts[1]  # dialect label
        
        # Map the dialect to its group
        group = dialect_to_group.get(dialect, "Unknown")
        
        # Write the updated line to the output file
        outfile.write(f"{utt_id} {group}\n")

print(f"Updated utt2group file saved to {output_utt2group}")
