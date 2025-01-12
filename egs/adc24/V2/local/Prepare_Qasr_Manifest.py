#!/usr/bin/env python3
# Copyright    2024          (author: KHELFI Mohammed Akram)

import os
import xml.etree.ElementTree as ET
from pydub import AudioSegment
from lhotse import RecordingSet, SupervisionSet, Recording, SupervisionSegment
from tqdm import tqdm
from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo
from lhotse.audio import AudioSource

import logging
from hyperion.hyp_defs import config_logger


def prepare_qasr(src_dir, seg_out_dir, dst_dir, verbose):
    config_logger(verbose)
    logging.info("Preparing QASR dataset")
    wav_dir = os.path.join(src_dir, "wav")
    xml_dir = os.path.join(src_dir, "release/train_20210109/xml")
    os.makedirs(dst_dir, exist_ok=True)

    recordings = []
    supervisions = []
    counter = 0

    for xml_file in tqdm(os.listdir(xml_dir)):
        if not xml_file.endswith(".xml"):
            continue
        logging.info(f"Processing {xml_file}")
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # get audio metadata
        head = root.find("head/recording")
        audio_file = f"{head.attrib['filename']}.wav"
        audio_path = os.path.join(wav_dir, audio_file)
        logging.info(f"Processing audio file {audio_file}")

        if not os.path.exists(audio_path):
            logging.error(f"Audio file {audio_path} not found")
            continue
        
        audio = AudioSegment.from_file(audio_path)

        speaker_map = {}
        for speaker in root.find("head/speakers"):
            speaker_id = speaker.attrib["id"]
            normalized_name = speaker.attrib.get("normalizedName", "")
            speaker_gender = speaker.attrib.get("speakerGender", "")
            speaker_map[speaker_id] = {"name": normalized_name, "gender": speaker_gender}

        for segment in root.find("body/segments"):
            segment_id = segment.attrib["id"]
            start_time = float(segment.attrib["starttime"]) * 1000 
            end_time = float(segment.attrib["endtime"]) * 1000
            speaker_id = segment.attrib["who"]
            text = " ".join([el.text for el in segment.findall("element")])

            segment_audio = audio[start_time:end_time]
            segment_audio = segment_audio.set_frame_rate(16000)
            segment_filename = f"{segment_id}.wav"
            segment_filepath = os.path.join(seg_out_dir, segment_filename)
            if not os.path.exists(segment_filepath):
                segment_audio.export(segment_filepath, format="wav")
            # else:
            #     logging.info(f"Segment file {segment_filename} already exists. Skipping.")

            recordings.append(
                Recording(
                id=segment_id,
                sources=[
                    AudioSource(
                        type="file",
                        channels=[0],
                        source=segment_filepath
                    )
                ],
                sampling_rate=16000,
                num_samples=len(segment_audio.get_array_of_samples()),
                duration=len(segment_audio) / 1000,
                )
            )


            speaker_data = speaker_map.get(speaker_id, {})
            supervisions.append(
                SupervisionSegment(
                    id=segment_id,
                    recording_id=segment_id,
                    start=0.0,
                    duration=(end_time - start_time) / 1000,
                    channel=0,
                    speaker=speaker_data.get("name", "unknown"),
                    custom={
                        "speaker_id": speaker_id,
                        "speaker_gender": speaker_data.get("gender", "unknown"),
                    },
                    text=text,
                    language="arabic",
                    )
            )
        counter += 1

    logging.info(f"Processed {counter} files")

    logging.info("Saving recordings and supervisions")
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    recording_set.to_file(os.path.join(dst_dir, "recordings.jsonl.gz"))
    supervision_set.to_file(os.path.join(dst_dir, "supervisions.jsonl.gz"))



  
    


if __name__ == "__main__":
  
    parser = ArgumentParser()
    parser.add_argument("--src-dir", type=str, required=True, help="source directory")
    parser.add_argument("--seg-out-dir", type=str, required=True, help="segmentation output directory")
    parser.add_argument("--dst-dir", type=str, required=True, help="destination directory")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)

    args = parser.parse_args()
    config_logger(args.verbose)
    prepare_qasr(**namespace_to_dict(args))
 