# TODO:
# evaluate:
# generate.generate_masked_song(songs[7], net, chord_map, chord_map_inv, pattern_map, pattern_map_inv, config, "test.mid", mask_start=8, mask_end=-8, tempo=120, time_sig=3, n_sample=20, max_length=64)
# ask: 1) how likely do you think it is that this melody was computer generated; 2) how interesting is this melody?

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import time
import argparse
import os
import json
import torch
import pickle
import numpy as np
from transformers import AdamW
import wandb

from datatypes import Chord
from config import config, HuggingFaceConfig
import model
import data
from device import device


def train():
    if config.gpu and False:
        from apex import amp

    print("loading data")
    train_songs = data.load_preprocessed_songs(limit=config.n_train)
    val_songs = data.load_preprocessed_songs(limit=config.n_val, offset=config.n_train)

    chord_map = data.load_chord_map()
    pattern_map = data.load_pattern_map()

    train_dl = data.data_loader(train_songs, chord_map, pattern_map, shuffle=True)
    val_dl = data.data_loader(val_songs, chord_map, pattern_map, shuffle=False)

    if os.path.exists("pretrained") and config.continue_from_checkpoint:
        net = model.LeadSheetForMaskedLM.from_pretrained("pretrained").to(device())
    else:
        net = model.LeadSheetForMaskedLM(HuggingFaceConfig()).to(device())

    optimizer = AdamW(net.parameters(), lr=config.learning_rate, correct_bias=True)

    if config.gpu and False:
        net, optimizer = amp.initialize(net, optimizer, opt_level=config.fp16_opt_level)

    wandb.init(project="music-inpainting")

    running_loss = []
    iteration = 0
    print("starting to train")
    for epoch in range(1000):
        net.train()

        for i, batch in enumerate(train_dl):
            optimizer.zero_grad()
            outputs = net(
                pattern_ids=batch["patterns"].to(device()),
                chord_ids=batch["chords"].to(device()),
                bar_numbers=batch["bar_numbers"].to(device()),
                beat_numbers=batch["beat_numbers"].to(device()),
                masked_pattern_labels=batch["pattern_labels"].to(device()),
                masked_chord_labels=batch["chord_labels"].to(device()),
                attention_mask=batch["attention_mask"].to(device()),
            )
            loss = outputs[0]

            if iteration % 1 == 0:
                print(iteration, loss.item())

            running_loss.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            if iteration % config.log_every == 0:
                print("logging")
                mean_loss = np.mean(running_loss)
                print("loss:", mean_loss)
                running_loss = []

                wandb.log(
                    {"loss": mean_loss},
                    step=iteration,
                )

            if iteration % config.checkpoint_every == 0:
                print("checkpointing")

                if not os.path.exists("pretrained"):
                    os.makedirs("pretrained")
                net.save_pretrained("pretrained")

            iteration += 1


if __name__ == "__main__":
    config.parse_args()
    train()
