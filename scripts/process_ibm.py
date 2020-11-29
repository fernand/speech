import os
import re
import sys

import sox
import scipy.io.wavfile

NON_ALPHA_QUOTE_REGEXP = re.compile(r"[^a-z\'\s]")
MULTI_SPACE_REGEXP = re.compile(r"\s+")

OUTPUT_DIR = "/data/ibm"


def split_write_audio_text(audio_f, transcripts):
    sr, y = scipy.io.wavfile.read(audio_f)
    assert sr == 16000
    for i, transcript in enumerate(transcripts):
        start, end = transcript[0], transcript[1]
        tfm = sox.Transformer()
        tfm.trim(start, end)
        audio_output_f = os.path.join(
            OUTPUT_DIR, os.path.basename(audio_f).split(".")[0] + f"_{i}.wav"
        )
        tfm.build_file(input_array=y, sample_rate_in=sr, output_filepath=audio_output_f)
        with open(audio_output_f.replace(".wav", ".txt"), "w") as f:
            f.write(transcript[2])


def get_transcripts(audio_path):
    transcript_path = audio_path.replace("wav.downsampled", "trs").replace(
        ".wav", ".trs"
    )
    at_turn = False
    after_turn = False
    last_sync = None
    current_sync = None
    transcript = None
    transcripts = []
    with open(transcript_path) as f:
        for l in f:
            if l.startswith("<Turn"):
                at_turn = True
            elif l.startswith("</Turn"):
                after_turn = True
            elif after_turn:
                continue
            elif not at_turn:
                continue
            if l.startswith("<Sync"):
                if last_sync is None:
                    last_sync = parse_sync(l)
                else:
                    current_sync = parse_sync(l)
                    if len(transcript) > 0:
                        transcripts.append((last_sync, current_sync, transcript))
                    last_sync = current_sync
            elif last_sync is not None:
                transcript = clean_transcript(l)
    return transcripts


def clean_transcript(transcript):
    transcript = transcript.strip().lower()
    transcript = re.sub(NON_ALPHA_QUOTE_REGEXP, "", transcript)
    transcript = re.sub(MULTI_SPACE_REGEXP, " ", transcript).lstrip()
    return transcript.strip()


def parse_sync(line):
    return float(line.split('"')[1])


if __name__ == "__main__":
    audio_paths = []
    with open("scripts/ibm_test.txt") as f:
        for l in f:
            audio_paths.append(l.strip())
    for audio_path in audio_paths:
        transcripts = get_transcripts(audio_path)
        split_write_audio_text(audio_path, transcripts)
