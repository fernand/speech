import os
import re
import itertools
import tempfile

import aeneas.task
import aeneas.executetask
import scipy.io.wavfile

TAG_REGEXP = re.compile(r"<.*?>")
NAME_REGEXP = re.compile(r" ([:alpha:]\:)|(\([:alpha:]\))")
MULTI_SPACE_REGEXP = re.compile(r"\s+")

# 01:02:51,435 --> 01:02:53,635
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
        if i > 0 and len(current_lines.intersection(prev_lines)) > 0:
            transcripts.pop()
            continue
        if "[" in transcript or ">>" in transcript:
            continue
        transcript = transcript.lower()
        transcript = re.sub(MULTI_SPACE_REGEXP, " ", transcript)
        transcript = re.sub(TAG_REGEXP, "", transcript)
        transcript = re.sub(NAME_REGEXP, "", transcript)
        transcript = transcript.replace('"', "")
        transcripts.append((start, end, transcript))
        prev_lines = current_lines
    f.close()
    return transcripts


def align_audio(transcripts, audio_f):
    tmp_txt = tempfile.NamedTemporaryFile(delete=False)
    for t in transcripts:
        tmp_txt.write((t[2] + "\n").encode("utf-8"))
    tmp_txt.close()
    config = "task_language=en|is_text_type=plain|os_task_file_format=txt"
    task = aeneas.task.Task(config_string=config)
    task.audio_file_path_absolute = audio_f
    task.text_file_path_absolute = tmp_txt.name
    aeneas.executetask.ExecuteTask(task).execute()
    res = task.sync_map_leaves()
    for fragment in res:
        print(fragment, fragment.text_fragment)
    os.remove(tmp_txt.name)


def split_audio(audio_f, transcripts):
    sr, wav = scipy
    assert sr == 16000


if __name__ == "__main__":
    f_id = "0_4_93"
    audio_f = f"/home/fernand/Downloads/{f_id}.wav"
    transcript_f = f"/home/fernand/Downloads/{f_id}.srt"
    transcripts = parse_srt(transcript_f)
    align_audio(transcripts, audio_f)
