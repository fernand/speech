import os
import pickle
import shutil
import sys
import time

import torch
import torchaudio
import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from decoder import cer, wer
from silero import prepare_model_input, load_silero_model, wav_to_text

torchaudio.set_audio_backend("soundfile")


class WavDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, input_dir):
        self.audio_files = audio_files
        self.input_dir = input_dir

    def __len__(self):
        return len(audio_files)

    def __getitem__(self, i):
        audio_f = os.path.join(self.input_dir, self.audio_files[i])
        try:
            wav, sr = torchaudio.load(audio_f, normalization=True, channels_first=True)
            wav = wav.squeeze(0)
        except:
            wav = torch.FloatTensor([])
        duration = len(wav) / 16000
        if duration > 6.0:
            wav = torch.FloatTensor([])
        transcript_f = audio_f.strip(".wav") + ".txt"
        transcript = None
        if os.path.exists(transcript_f):
            with open(transcript_f, "r") as f:
                transcript = f.read().strip()
                if len(transcript) == 0:
                    transcript = None
        return (audio_f, wav, transcript, duration)


def collate_fn(data):
    return (
        [t[0] for t in data],
        prepare_model_input([t[1] for t in data], device=torch.device("cpu")),
        [t[2] for t in data],
        [t[3] for t in data],
    )


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    device = torch.device(f"cuda:1")
    model, decoder = load_silero_model(device)
    dataset = WavDataset(audio_files, input_dir)
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=3,
        pin_memory=True,
    )

    manifest = []
    for batch in tqdm.tqdm(dataloader, total=len(dataset) // batch_size):
        files, wavs, transcripts, durations = batch
        wavs = wavs.to(device)
        output = model(wavs).cpu()
        predictions = [decoder(output[i]) for i in range(len(output))]
        for audio_f, transcript, prediction, duration in zip(
            files, transcripts, predictions, durations
        ):
            if transcript is None:
                continue
            if len(prediction.strip()) == 0:
                char_error = 1.0
                word_error = 1.0
            else:
                char_error = cer(transcript, prediction)
                word_error = wer(transcript, prediction)
            manifest.append((audio_f, char_error, word_error, duration))

    with open(os.path.join(output_dir, f"manifest.pkl"), "wb") as f:
        pickle.dump(manifest, f)
