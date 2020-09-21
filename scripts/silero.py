import sys

import torch
from omegaconf import OmegaConf

sys.path.append("silero-models/")
import utils

models = OmegaConf.load("./silero-models/models.yml")
device = torch.device("cpu")
model, decoder = utils.init_jit_model(models.stt_models.en.latest.jit, device=device)


def wav_to_text(f):
    batch = utils.read_batch([f])
    input = utils.prepare_model_input(batch, device=device)
    output = model(input)
    return decoder(output[0].cpu())

