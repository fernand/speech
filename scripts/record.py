import collections
import os
import signal
import subprocess
import sys
import time

REC_LEN_S = 60 * 60

# sudo chmod -R 777 /dev/dvb/adapter0
# tsscan --verbose --hf-band-region us --vhf-band -a 0 --save-channels channels.xml
# tsscan --verbose --hf-band-region us --uhf-band -a 0 --save-channels channels2.xml
FREQUENCIES = collections.OrderedDict(
    {
        "177,000,000": [3, 4, 5, 6],
        "575,000,000": [3, 4, 5],
        "605,000,000": [3, 4, 5],
        "599,000,000": [1, 2, 3],
    }
)


def clip_name(clip_i, adapter):
    if clip_i % 2 == 0:
        clip_name = f"clip{adapter}_0.ts"
    else:
        clip_name = f"clip{adapter}_1.ts"
    return clip_name


def base_name(clip_i, adapter, program):
    return f"{adapter}_{program}_{clip_i}"


def start_recording(clip_i, adapter, frequency):
    tsp = [
        "tsp",
        "-I",
        "dvb",
        "--device-name",
        f"/dev/dvb/adapter{adapter}",
        "--frequency",
        frequency,
        "--modulation",
        "8-VSB",
        "-O",
        "file",
        clip_name(clip_i, adapter),
    ]
    return subprocess.Popen(tsp, stderr=subprocess.PIPE)


def extract_subtitles(clip_i, adapter, program):
    ccextractor = [
        "ccextractor",
        "-quiet",
        "-pn",
        program,
        clip_name(clip_i, adapter),
        "-o",
        f"{base_name(clip_i, adapter, program)}.srt",
    ]
    p = subprocess.Popen(ccextractor, stderr=subprocess.PIPE)
    _, stderr = p.communicate()
    if len(stderr) > 0:
        print(stderr)


def extract_audio(clip_i, adapter, program):
    ffmpeg = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "panic",
        "-i",
        clip_name(clip_i, adapter),
        "-map",
        f"0:p:{program}:1",
        "-c",
        "copy",
        f"{base_name(clip_i, adapter, program)}.ac3",
    ]
    p = subprocess.Popen(ffmpeg, stderr=subprocess.PIPE)
    _, stderr = p.communicate()
    if len(stderr) > 0:
        print(stderr)


def recording_loop(clip_i, adapter, frequency, programs):
    tsp_p = start_recording(clip_i, adapter, frequency)
    try:
        while True:
            time.sleep(REC_LEN_S)
            prev_clip = clip_i
            clip_i += 1
            tsp_p.send_signal(signal.SIGINT)
            time.sleep(2)
            tsp_p = start_recording(clip_i, adapter, frequency)
            for program in programs:
                extract_subtitles(prev_clip, adapter, program)
                srt_file = base_name(prev_clip, adapter, program) + ".srt"
                if not os.path.exists(srt_file):
                    continue
                with open(srt_file, "r") as f:
                    subtitles = f.readlines()
                if len(subtitles) > 2:
                    extract_audio(prev_clip, adapter, program)
                else:
                    os.remove(srt_file)
            os.remove(clip_name(prev_clip, adapter))
    except KeyboardInterrupt:
        tsp_p.send_signal(signal.SIGINT)
        sys.exit(0)


if __name__ == "__main__":
    adapter = int(sys.argv[1])
    frequency, programs = list(FREQUENCIES.items())[adapter]
    recording_loop(108, adapter, frequency, [str(p) for p in programs])
