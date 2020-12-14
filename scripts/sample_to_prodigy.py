import json
import os
import pickle
import sys
import random

# Update prodigy.json to disable cors and set host to the right IP.
# prodigy audio.transcribe utterances audio/utterances.jsonl --loader jsonl
# cd /; python -m http.server 8081 --bind 192.168.1.21
# Use Safari and disable CORS.
def to_jsonl(utterances, output_f):
    if os.path.exists(output_f):
        os.remove(output_f)
    random.shuffle(utterances)
    with open(output_f, "w") as f:
        for utterance in utterances:
            url = "http://192.168.1.21:8081" + utterance[0]
            js = {"audio": url, "transcript": utterance[1]}
            f.write(json.dumps(js) + "\n")


if __name__ == "__main__":
    cer = float(sys.argv[1])
    output_json_f = sys.argv[2]
    num_to_sample = 100
    datasets = ["first", "fourth"]
    utterances = []
    for dataset in [f"datasets/{d}/manifest.pkl" for d in datasets]:
        with open(dataset, "rb") as f:
            m = pickle.load(f)
        candidates = [t[0] for t in m if t[1] >= cer and t[1] < cer + 0.005]
        random.shuffle(candidates)
        samples = candidates[:num_to_sample]
        for path in samples:
            path = path.replace("/data", "/hd1")
            text_f = path.replace(".wav", ".txt")
            with open(text_f) as f:
                utterance = f.read().strip()
                utterances.append((path, utterance))
    to_jsonl(utterances, output_json_f)
