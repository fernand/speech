import os
import sys

import torch
from omegaconf import OmegaConf

REPO_DIR = "/home/fernand/dependencies/silero-models/"

sys.path.append(REPO_DIR)
import utils


def load_silero_model():
    device = torch.device("cpu")
    torch.set_num_threads(1)
    models = OmegaConf.load(os.path.join(REPO_DIR, "models.yml"))
    model, decoder = utils.init_jit_model(
        models.stt_models.en.latest.jit, device=device
    )
    return model, decoder


def wav_to_text(f, model, decoder):
    batch = utils.read_batch([f])
    inp = utils.prepare_model_input(batch, device=torch.device("cpu"))
    output = model(inp)
    return decoder(output[0].cpu())
