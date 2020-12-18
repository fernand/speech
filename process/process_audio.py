import itertools
import multiprocessing
import os
import pickle
import re
import random
import subprocess
import sys
import tempfile
import time
import uuid

import aeneas.task
import aeneas.executetask
import scipy.io.wavfile
import sox
import numpy as np

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
    transcript = re.sub(NON_ALPHA_QUOTE_REGEXP, "", transcript)
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


def get_transcript_chunks(transcripts):
    chunks = []
    first_tr = transcripts[0]
    current_chunk = [first_tr]
    prev_end = first_tr[1]
    for tr in transcripts[1:]:
        current_start = tr[0]
        if current_start - prev_end < 0.5:
            current_chunk.append(tr)
        else:
            chunks.append(current_chunk)
            current_chunk = [tr]
        prev_end = tr[1]
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
    assert sr == 16000
    outputs = []
    for i, chunk in enumerate(transcript_chunks):
        if len(chunk) <= 3:
            continue
        start, end = chunk[0][0], chunk[-1][1]
        tfm = sox.Transformer()
        tfm.trim(start, end)
        output_f = os.path.join(
            output_dir, os.path.basename(audio_f).split(".")[0] + f"_{i}.wav"
        )
        tfm.build_file(input_array=y, sample_rate_in=sr, output_filepath=output_f)
        outputs.append((output_f, "\n".join([tr[2] for tr in chunk])))
    return outputs


def align_audio(audio_f, transcript_lines):
    tmp_txt = tempfile.NamedTemporaryFile(delete=False)
    tmp_txt.write(transcript_lines.encode("utf-8"))
    tmp_txt.close()
    config = "task_language=en|is_text_type=plain|os_task_file_format=txt"
    task = aeneas.task.Task(config_string=config)
    task.audio_file_path_absolute = audio_f
    task.text_file_path_absolute = tmp_txt.name
    aeneas.executetask.ExecuteTask(task).execute()
    os.remove(tmp_txt.name)
    return task.sync_map


def split_chunk_to_uterances(audio_f, fragments, output_dir):
    sr, y = scipy.io.wavfile.read(audio_f)
    assert sr == 16000
    for i, fragment in enumerate(fragments):
        start, end = float(str(fragment.begin)), float(str(fragment.end))
        # Ignore utterances shorter than 1 second
        if end < start + 1.0:
            continue
        tfm = sox.Transformer()
        tfm.trim(start, end)
        output_f = os.path.join(
            output_dir, os.path.basename(audio_f).split(".")[0] + f"_{i}.wav"
        )
        tfm.build_file(input_array=y, sample_rate_in=sr, output_filepath=output_f)
        output_txt = output_f.strip(".wav") + ".txt"
        with open(output_txt, "w") as txt_f:
            txt_f.write(fragment.text + "\n")


def opusenc(audio_f):
    assert audio_f.endswith(".wav")
    opus = ["opusenc", "--quiet", audio_f, audio_f.strip(".wav") + ".opus"]
    p = subprocess.Popen(opus)
    p.communicate()
    os.remove(audio_f)


def process_file(audio_f, output_dir):
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
    for audio_chunk_f, transcript in outputs:
        # Make sure the wavfile is valid.
        sr, y = scipy.io.wavfile.read(audio_chunk_f)
        if len(y) == 0:
            os.remove(audio_chunk_f)
            continue
        sync_map = align_audio(audio_chunk_f, transcript)
        fragments = sync_map.leaves(fragment_type=0)
        split_chunk_to_uterances(audio_chunk_f, fragments, output_dir)
        os.remove(audio_chunk_f)


DATASETS = {
    "first": {
        "input_dirs": ["/hd1/first", "/hd1/first/extra", "/hd1/first/round1"],
        "output_dir": "/hd1/clean",
    },
    "second": {
        "input_dirs": ["/hd1/second", "/hd1/second/first", "/hd1/second/second"],
        "output_dir": "/hd1/clean2",
    },
    "third": {"input_dirs": ["/hd1/third"], "output_dir": "/hd1/clean3"},
    "fourth": {"input_dirs": ["/hd1/fourth"], "output_dir": "/hd1/clean4"},
}

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    input_dirs = DATASETS[dataset_name]["input_dirs"]
    output_dir = DATASETS[dataset_name]["output_dir"]
    audio_files = list_input_audio_files(input_dirs)
    random.shuffle(audio_files)
    print(f"{len(audio_files)} files to process")
    # Process by chunks in order to not run into RAM issues.
    num_chunks = 20
    chunks = np.array_split(audio_files, num_chunks)
    for chunk_i, chunk in enumerate(chunks):
        p = multiprocessing.Pool(32)
        print(f"Processing chunk {chunk_i} out of {num_chunks - 1}")
        start = time.time()
        p.starmap(process_file, [(audio_f, output_dir) for audio_f in chunks[chunk_i]])
        print(f"Time to process: {round(time.time() - start)}s")
        p.close()
        p.join()
