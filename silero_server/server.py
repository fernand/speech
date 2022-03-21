"""
gunicorn -w 2 -b 127.0.0.1:4000 "server:app"
"""
import ujson
from flask import Flask, request

import silero_utils

app = Flask(__name__)
model, decoder, device_id = None, None, None


@app.route("/align", methods=["POST"])
def align():
    payload = ujson.loads(request.json)
    audio_paths = payload["audio_paths"]
    batch_size = payload["batch_size"]
    global model, decoder, device_id
    if model is None:
        (model, decoder, _), device_id = silero_utils.get_model_tuple()

    result = {}
    batches = silero_utils.split_into_batches(audio_paths, batch_size=batch_size)
    file_idx = 0
    for batch in batches:
        loaded_batch = silero_utils.read_batch(batch)
        input = silero_utils.prepare_model_input(loaded_batch, device=device_id)
        output = model(input)
        for i, example in enumerate(output):
            alignment = decoder(
                example.cpu(), wav_len=len(loaded_batch[i]) / 16000, word_align=True
            )
            result[audio_paths[file_idx]] = alignment
            file_idx += 1

    return result


if __name__ == "__main__":
    app.run()
