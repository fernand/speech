import os
import re
import itertools
import tempfile

import aeneas.task
import aeneas.executetask
import scipy.io.wavfile
import sox

TAG_REGEXP = re.compile(r"<.*?>")
NAME_REGEXP = re.compile(r"([A-z]+\:)|(\([A-z]+\))")
PUNCT_REGEXP = re.compile(r"[\,\.\-\"!]")
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
        transcript = re.sub(TAG_REGEXP, "", transcript)
        transcript = re.sub(NAME_REGEXP, "", transcript)
        transcript = re.sub(PUNCT_REGEXP, "", transcript)
        transcript = re.sub(MULTI_SPACE_REGEXP, " ", transcript).lstrip()
        transcripts.append((start, end, transcript))
        prev_lines = current_lines
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


def split_audio(audio_f, transcript_chunks, output_dir):
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


if __name__ == "__main__":
    f_id = "0_4_93"
    output_dir = "/home/fernand/Downloads"
    audio_f = f"/home/fernand/Downloads/{f_id}.wav"
    transcript_f = f"/home/fernand/Downloads/{f_id}.srt"
    transcripts = parse_srt(transcript_f)
    chunks = get_transcript_chunks(transcripts)
    # with open("transcript.txt", "w") as f:
    #     f.write("\n".join([str(tr) for tr in transcripts]))
    outputs = split_audio(audio_f, chunks, output_dir)
    for abs_start, audio_chunk_f, transcript in outputs:
        sync_map = align_audio(audio_chunk_f, transcript)
        fragments = sync_map.leaves(fragment_type=0)
        for fragment in fragments:
            print(abs_start, fragment, fragment.text_fragment)
        print("======================================")
