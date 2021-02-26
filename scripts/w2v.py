import os
import sys

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import soundfile as sf
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import data
import silero_utils
from decoder import cer, wer

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(
    "cuda"
)
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


def cfn(data):
    return tokenizer([t[0] for t in data], return_tensors="pt", padding="longest"), [
        t[1] for t in data
    ]


test_loader = torch.utils.data.DataLoader(
    dataset=data.IBMDataset(),
    batch_size=32,
    shuffle=False,
    collate_fn=cfn,
    num_workers=3,
    pin_memory=True,
)

test_cer, test_wer = [], []
for batch in test_loader:
    inputs, transcripts = batch
    input_values = inputs["input_values"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predictions = tokenizer.batch_decode(predicted_ids)
    for transcript, prediction in zip(transcripts, predictions):
        prediction = prediction.lower()
        if len(prediction.strip()) == 0:
            char_error = 1.0
            word_error = 1.0
        else:
            char_error = cer(transcript, prediction)
            word_error = wer(transcript, prediction)
        test_cer.append(char_error)
        test_wer.append(word_error)

avg_cer = sum(test_cer) / len(test_cer)
avg_wer = sum(test_wer) / len(test_wer)
print("Average CER: {:4f} Average WER: {:.4f}\n".format(avg_cer, avg_wer))
