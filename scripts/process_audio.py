import hashlib
import itertools
import json
import multiprocessing
import os
import pickle
import re
import random
import subprocess
import tempfile

import aeneas.task
import aeneas.executetask
import scipy.io.wavfile
import sox

TAG_REGEXP = re.compile(r"<.*?>")
NAME_REGEXP = re.compile(r"([A-z]+\:)|(\([A-z]+\))")
NON_ALPHA_QUOTE_REGEXP = re.compile(r"[^a-z\'\s]")
MULTI_SPACE_REGEXP = re.compile(r"\s+")


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
    seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    return seconds


def parse_srt(srt_f):
    transcripts = []
    f = open(srt_f)
    lines = f.readlines()
    chunks = [
        tuple(y) for x, y in itertools.groupby(lines, lambda z: z == "\n") if not x
    ]
    prev_lines = set()
    current_lines = set()
    for i, chunk in enumerate(chunks):
        start_end = chunk[1].strip().split(" --> ")
        start, end = map(time_to_seconds, start_end)
        transcript = " ".join([c.strip() for c in chunk[2:]])
        current_lines = set(chunk[2:])
        # Ignore transcripts which overlap on multiple positions.
        if len(current_lines.intersection(prev_lines)) > 0:
            if len(transcripts) > 0:
                transcripts.pop()
            prev_lines = current_lines
            continue
        prev_lines = current_lines
        if "[" in transcript or ">>" in transcript:
            continue
        transcript = transcript.lower()
        transcript = re.sub(TAG_REGEXP, "", transcript)
        transcript = re.sub(NAME_REGEXP, "", transcript)
        transcript = re.sub(NON_ALPHA_QUOTE_REGEXP, "", transcript)
        transcript = re.sub(MULTI_SPACE_REGEXP, " ", transcript).lstrip()
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
    f_id = audio_f.strip(".ac3")
    f_hash = hashlib.sha1(f_id.encode("utf-8")).hexdigest()
    output_f = os.path.join(output_dir, f_hash + ".wav")
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
    _, stderr = p.communicate()
    if len(stderr) > 0:
        print(audio_f, stderr)
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
        outputs.append((chunk[0][0], output_f, "\n".join([tr[2] for tr in chunk])))
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
    outputs = []
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
        outputs.append((output_f, fragment.text, end - start))
    return outputs


# prodigy audio.transcribe uterances audio/uterances.jsonl --loader jsonl
# python -m http.server 8081 --bind 192.168.1.21
def to_jsonl(uterances, output_f):
    if os.path.exists(output_f):
        os.remove(output_f)
    random.shuffle(uterances)
    with open(output_f, "w") as f:
        for uterance in uterances:
            file = os.path.basename(uterance[0])
            url = "http://192.168.1.21:8081/" + file
            js = {"audio": url, "transcript": uterance[1]}
            f.write(json.dumps(js) + "\n")


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
    utterances = []
    for _, audio_chunk_f, transcript in outputs:
        sync_map = align_audio(audio_chunk_f, transcript)
        fragments = sync_map.leaves(fragment_type=0)
        utterances.extend(
            split_chunk_to_uterances(audio_chunk_f, fragments, output_dir)
        )
        os.remove(audio_chunk_f)
    return audio_f, utterances


if __name__ == "__main__":
    input_dirs = ["/tv/first", "/tv/first/extra", "/tv/first/round1"]
    output_dir = "/data/clean"
    audio_files = list_input_audio_files(input_dirs)
    p = multiprocessing.Pool(6)
    res = p.starmap(process_file, [(f, output_dir) for f in audio_files])
    res = [t for t in res if t != ()]
    with open("/data/clean/manifest.pkl", "wb") as f:
        pickle.dump(res, f)
