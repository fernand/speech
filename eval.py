import html
import time
from html.parser import HTMLParser

import scipy.io.wavfile
import torch

import net
import data


def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# example for s: 01:02:51.435
def time_to_seconds(s):
    h, m, sm = s.split(":")
    s, ms = sm.split(".")
    if any([len(x) == 0 for x in [h, m, s, ms]]):
        return None
    seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    return seconds


class TTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.times = []
        self.transcripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            d = dict(attrs)
            self.times.append((time_to_seconds(d["begin"]), time_to_seconds(d["end"])))

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        self.transcripts.append(html.unescape(data))


if __name__ == "__main__":
    parser = TTMLParser()
    with open("/home/fernand/subtitle.ttml", "rt") as f:
        parser.feed(f.read())
    sr, wav = scipy.io.wavfile.read("/home/fernand/interview.wav")
    wav_pieces = []
    for i in range(1, len(parser.times) - 1):
        wav_pieces.append(
            wav[
                int(16000 * parser.times[i - 1][0]) : int(
                    16000 * parser.times[i + 1][0]
                )
            ]
        )
    checkpoint = torch.load("model_1.pth")
    model = net.Model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.eval().cuda()

    chunks = get_chunks(wav_pieces, 256)
    predictions = []
    for chunk in chunks:
        spectrograms = []
        for piece in chunk:
            spectrograms.append(
                data.spectrogram_transform(torch.from_numpy(piece / 32767).float())
                .squeeze(0)
                .transpose(0, 1)
            )
        spectrograms = torch.nn.utils.rnn.pad_sequence(
            spectrograms, batch_first=True
        )  # B, T, C
        spectrograms = (
            spectrograms.unsqueeze(1).transpose(2, 3).half().cuda()
        )  # B, 1, C, T

        with (torch.no_grad(), torch.cuda.amp.autocast()):
            output = model(spectrograms).sigmoid().cpu().numpy()
        predictions.extend(output.tolist())
