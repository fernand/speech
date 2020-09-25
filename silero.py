import os
import sys

import torch
from omegaconf import OmegaConf

REPO_DIR = "/home/fernand/dependencies/silero-models/"

sys.path.append(REPO_DIR)
import utils

prepare_model_input = utils.prepare_model_input


def load_silero_model(device):
    models = OmegaConf.load(os.path.join(REPO_DIR, "models.yml"))
    model, decoder = utils.init_jit_model(
        models.stt_models.en.latest.jit, device=device
    )
    return model, decoder


def wav_to_text(f, model, decoder, device):
    batch = utils.read_batch([f])
    inp = utils.prepare_model_input(batch, device=device)
    output = model(inp)
    return decoder(output[0].cpu())
