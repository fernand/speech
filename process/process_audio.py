"""
WIP. Need to calculate alignment with biopython/alignment then write the wav files.
"""
import itertools
import math
import multiprocessing
import os
import re
import random
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import scipy.io.wavfile
import torch

import int_to_words


def list_input_audio_files(input_dirs):
    files = []
    for input_dir in input_dirs:
        dir_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".ac3")
        ]
        files.extend(dir_files)
    return files


# example for s: 01:02:51,435
def time_to_seconds(s):
    h, m, sm = s.split(":")
    s, ms = sm.split(",")
    if any([len(x) == 0 for x in [h, m, s, ms]]):
        return None
    seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    return seconds


CONTINUATION_REGEXP_1 = re.compile(r">>")
CONTINUATION_REGEXP_2 = re.compile(r">>>")
TAG_REGEXP = re.compile(r"<.*?>")
NAME_REGEXP = re.compile(r"([A-z]+\:)|(\([A-z]+\))")
NUMBER_REGEXP = re.compile(r"\d+")
ACCENT_REGEXP = re.compile(r"[éèêëà]")
DASH_REGEXP = re.compile(r"-")
NON_ALPHA_QUOTE_REGEXP = re.compile(r"[^a-z\'\s]")
MULTI_SPACE_REGEXP = re.compile(r"\s+")


def replace_num(matchobj):
    return " " + int_to_words.name_number(int(matchobj.group(0))) + " "


ACCENT_DICT = {"é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a"}

# Ignore the transcript if those symbols are included.
BLACKLIST = set(["[", "$", "¢", ".com"])


def remove_accent(matchobj):
    return ACCENT_DICT[matchobj.group(0)]


def clean_transcript(transcript):
    transcript = transcript.lower()
    transcript = re.sub(CONTINUATION_REGEXP_1, "", transcript)
    transcript = re.sub(CONTINUATION_REGEXP_2, "", transcript)
    transcript = re.sub(TAG_REGEXP, "", transcript)
    transcript = re.sub(NAME_REGEXP, "", transcript)
    transcript = re.sub(NUMBER_REGEXP, replace_num, transcript)
    transcript = re.sub(ACCENT_REGEXP, remove_accent, transcript)
    transcript = re.sub(DASH_REGEXP, " ", transcript)
    transcript = re.sub(NON_ALPHA_QUOTE_REGEXP, " ", transcript)
    transcript = re.sub(MULTI_SPACE_REGEXP, " ", transcript).lstrip()
    return transcript


def parse_srt(srt_f):
    transcripts = []
    f = open(srt_f)
    lines = f.readlines()
    chunks = [
        tuple(y) for x, y in itertools.groupby(lines, lambda z: z == "\n") if not x
    ]
    prev_lines = []
    for i, chunk in enumerate(chunks):
        start_end = chunk[1].strip().split(" --> ")
        if len(start_end) != 2:
            continue
        start, end = map(time_to_seconds, start_end)
        if start is None or end is None:
            continue
        current_lines = chunk[2:]
        if len(current_lines) == 0:
            continue

        # For transcripts which overlap on multiple positions,
        # remove the second transcript occurence.
        filtered_lines = []
        for line in current_lines:
            if line not in prev_lines:
                filtered_lines.append(line)
        prev_lines = current_lines

        transcript = " ".join([l.strip() for l in filtered_lines])
        if any([c in transcript for c in BLACKLIST]):
            continue
        transcript = clean_transcript(transcript)
        if len(transcript) > 0:
            transcripts.append((start, end, transcript))
    f.close()
    return transcripts


# Split the transcript into 20 second chunks with one transcript overlap.
def get_transcript_chunks(transcripts):
    chunks = []
    first_tr = transcripts[0]
    current_chunk = [first_tr]
    current_chunk_start = first_tr[0]
    for tr in transcripts[1:]:
        current_end = tr[0]
        if current_end - current_chunk_start < 20:
            current_chunk.append(tr)
        else:
            chunks.append(current_chunk)
            first_tr = current_chunk[-1]
            current_chunk = [first_tr, tr]
            current_chunk_start = first_tr[0]
    chunks.append(current_chunk)
    return chunks


def ac3_to_wav(audio_f, output_dir):
    output_f = os.path.join(output_dir, str(uuid.uuid4()) + ".wav")
    ffmpeg = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "panic",
        "-i",
        audio_f,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        output_f,
    ]
    p = subprocess.Popen(ffmpeg, stderr=subprocess.PIPE)
    p.communicate()
    return output_f


def split_audio_to_chunks(audio_f, transcript_chunks, output_dir):
    sr, y = scipy.io.wavfile.read(audio_f)
    len_y = len(y)
    assert sr == 16000
    outputs = []
    for i, chunk in enumerate(transcript_chunks):
        if len(chunk) <= 3:
            continue
        start_s, end_s = chunk[0][0], chunk[-1][1]
        start = max(0, int(sr * start_s))
        end = min(len_y, math.ceil(sr * end_s))
        output_f = os.path.join(
            output_dir, os.path.basename(audio_f).split(".")[0] + f"_{i}.wav"
        )
        scipy.io.wavfile.write(output_f, sr, y[start:end])
        outputs.append((output_f, "\n".join([tr[2] for tr in chunk])))
    return outputs


def opusenc(audio_f):
    assert audio_f.endswith(".wav")
    opus = ["opusenc", "--quiet", audio_f, audio_f.strip(".wav") + ".opus"]
    p = subprocess.Popen(opus)
    p.communicate()
    os.remove(audio_f)


def process_file(audio_f, output_dir, model, decoder, model_utils):
    srt_f = audio_f.split(".")[0] + ".srt"
    transcripts = parse_srt(srt_f)
    if len(transcripts) == 0:
        return ()
    wav_f = ac3_to_wav(audio_f, output_dir)
    if not os.path.exists(wav_f):
        return ()
    chunks = get_transcript_chunks(transcripts)
    outputs = split_audio_to_chunks(wav_f, chunks, output_dir)
    os.remove(wav_f)

    audio_chunk_files = [o[0] for o in outputs]
    (read_batch, split_into_batches, read_audio, prepare_model_input) = model_utils
    batches = split_into_batches(audio_chunk_files, batch_size=32)

    silero_alignments = []
    file_idx = 0
    for batch in batches:
        loaded_batch = read_batch(batch)
        input = prepare_model_input(loaded_batch, device=device)
        output = model(input)
        for i, example in enumerate(output):
            alignment = decoder(
                example.cpu(), wav_len=len(loaded_batch[i]), word_align=True
            )
            print(batch[i], outputs[file_idx][1].replace("\n", " "))
            print(alignment)
            print("------------")
            silero_alignments.append(alignment)
            file_idx += 1
    print(audio_f)
    print("=====================================")

    # for i, t in enumerate(outputs):
    #     audio_chunk_f, transcripts = t
    #     alignment = silero_alignments[i]

    #     # Make sure the wavfile is valid.
    #     sr, y = scipy.io.wavfile.read(audio_chunk_f)
    #     if len(y) == 0:
    #         os.remove(audio_chunk_f)
    #         continue


DATASETS = {
    "first": {
        "input_dirs": ["/backup/first", "/backup/first/extra", "/backup/first/round1"],
        "output_dir": "/nvme/clean",
    },
    "second": {
        "input_dirs": [
            "/backup/second",
            "/backup/second/first",
            "/backup/second/second",
        ],
        "output_dir": "/nvme/clean2",
    },
    "third": {"input_dirs": ["/backup/third"], "output_dir": "/nvme/clean3"},
    "fourth": {"input_dirs": ["/backup/fourth"], "output_dir": "/nvme/clean4"},
    "fifth": {"input_dirs": ["/backup/fifth"], "output_dir": "/nvme/clean5"},
    "sixth": {"input_dirs": ["/backup/sixth"], "output_dir": "/nvme/clean6"},
}

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    input_dirs = DATASETS[dataset_name]["input_dirs"]
    output_dir = DATASETS[dataset_name]["output_dir"]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    audio_files = list_input_audio_files(input_dirs)
    random.shuffle(audio_files)
    print(f"{len(audio_files)} files to process")

    device = torch.device("cuda:0")
    model, decoder, model_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_stt",
        language="en",
        device=device,
    )

    # process_file("/backup/first/1_4_228.ac3", output_dir, model, decoder, model_utils)
    process_file(audio_files[0], output_dir, model, decoder, model_utils)

    # Process by chunks in order to not run into RAM issues.
    # num_chunks = 20
    # chunks = np.array_split(audio_files, num_chunks)
    # for chunk_i, chunk in enumerate(chunks):
    #     p = multiprocessing.Pool(32)
    #     print(f"Processing chunk {chunk_i} out of {num_chunks - 1}")
    #     start = time.time()
    #     p.starmap(process_file, [(audio_f, output_dir) for audio_f in chunks[chunk_i]])
    #     print(f"Time to process: {round(time.time() - start)}s")
    #     p.close()
    #     p.join()

a = "mary jenkins  madame you're a genius this work captures the frustration of the modern housewife  mary is he serious  seriously ill what else have you created  i created a daughter  but i had some help  i'm sorry i have to s ay i'm not an artist"
b = "deins mea you're a genius this work captures the frustgrace of the modern housewife mary is he serious seriously ill what have you created well i created a donllar but i had little i'm sorryi have to tell you i'm not in"
