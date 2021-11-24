import math
from fractions import Fraction
import numpy as np
import json
from glob import glob
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import music21 as m21

import chords
from datatypes import Song, Chord, Pattern
from config import config

BEAT_DIV = 240


def load_chord_map():
    with open("chord_map.json", "r") as f:
        chord_map_dicts = json.load(f)
    chord_map = {Chord.from_dict(c): int(i) + 1 for i, c in chord_map_dicts.items()}
    return chord_map


def load_pattern_map():
    with open("pattern_map.json", "r") as f:
        pattern_map_raw = json.load(f)
    pattern_map = {
        Pattern.from_list(pat): int(i) + 1 for i, pat in pattern_map_raw.items()
    }
    return pattern_map


def load_preprocessed_songs(limit=None, offset=None) -> List[Song]:
    songs: List[Song] = []

    paths = list(sorted(glob("preprocessed-json/*.json")))

    for i, path in enumerate(paths):
        if limit is not None and i == limit:
            break
        if i % 100 == 0:
            print(i)
        with open(path) as f:
            try:
                song = Song.from_dict(json.load(f), path)
            except Exception as e:
                print("Failed to read %s: %s" % (path, e))
                continue

            if len(song.notes) < 25:
                continue

            if len(song.chords) < 5:
                continue

            if song.chords[0][0] > 10:
                continue

            if len(song.bars) < 5:
                print("too few bars in %s: %d" % (path, len(song.bars)))
                continue

            if song.duration > 1000:
                print("too long %s: %d" % (path, song.duration))
                continue

            note_times = [t for t, n in song.notes]
            if note_times != list(sorted(note_times)):
                continue

            songs.append(song)

    return songs


def clear_tokens(
    inputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = inputs.clone()
    labels = inputs.clone()

    inputs[inputs != config.pad_token_id] = config.mask_token_id

    return inputs, labels


def mask_tokens(
    inputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = inputs.clone()
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, config.mask_prob)
    padding_mask = labels.eq(config.pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = config.mask_token_id

    return inputs, labels

    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(127, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels


class LeadSheetDataset(Dataset):
    def __init__(self, songs, chord_map, pattern_map):
        self.songs = songs
        self.chord_map = chord_map
        self.pattern_map = pattern_map

    def __len__(self):
        return len(self.songs) * 12

    def __getitem__(self, index):
        song = self.songs[index // 12]
        step = index % 12
        song = song.transpose(step)
        patches = song_to_arrays(
            song,
            chord_map=self.chord_map,
            pattern_map=self.pattern_map,
        )
        patterns, chords, bar_numbers, beat_numbers = patches[
            np.random.randint(len(patches))
        ]

        masked_patterns, pattern_labels = mask_tokens(patterns)
        masked_chords, chord_labels = mask_tokens(chords)

        attention_mask = (masked_patterns != config.pad_token_id).to(int)

        return {
            "patterns": masked_patterns,
            "chords": masked_chords,
            "bar_numbers": bar_numbers,
            "beat_numbers": beat_numbers,
            "pattern_labels": pattern_labels,
            "chord_labels": chord_labels,
            "attention_mask": attention_mask,
        }


def data_loader(songs, chord_map, pattern_map, shuffle):
    train_dl = DataLoader(
        LeadSheetDataset(songs, chord_map, pattern_map),
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=True,
        #num_workers=0,
    )
    return train_dl


def load_wjazzd() -> List[Song]:
    conn = sqlite3.connect("wjazzd.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    n_mels = 456

    mel_map = {}

    for i in range(1, n_mels + 1):
        cur.execute(
            "select melid, onset, bar, beat, division, tatum, tatumprops, pitch from melody where melid = %d order by onset"
            % i
        )
        rows = cur.fetchall()
        for r in rows:
            r = dict(r)
            key = (r["melid"], r["bar"], r["beat"], r["division"])
            mel_map[key] = int(round(r["pitch"]))

    songs: List[Song] = []
    for mel in range(1, n_mels + 1):
        cur.execute(
            "select onset, bar, beat, chord from beats where bar >= 0 and melid = %d order by onset"
            % mel
        )
        rows = cur.fetchall()
        chord = ""
        pitch = None
        song = Song(mel)
        for r in rows:
            r = dict(r)
            if r["chord"] != "":
                chord = r["chord"]

            for div in range(1, 5):
                bar = r["bar"]
                beat = r["beat"]
                key = (mel, bar, beat, div)
                if key in mel_map:
                    pitch = mel_map[key]

                song.append(
                    Div(
                        bar=r["bar"],
                        beat=r["beat"],
                        div=div,
                        pitch=pitch,
                        chord=chords.harte(chord).title,
                    )
                )

        songs.append(song)

    return songs


def song_to_arrays(
    song: Song,
    chord_map: Dict[Chord, int],
    pattern_map: Dict[Pattern, int],
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    patterns = torch.zeros(size=(math.ceil(song.duration),), dtype=torch.long)
    chords = (
        torch.zeros(size=(math.ceil(song.duration),), dtype=torch.long)
        + config.pad_token_id
    )
    bar_numbers = torch.zeros(size=(math.ceil(song.duration),), dtype=torch.long)
    beat_numbers = torch.zeros(size=(math.ceil(song.duration),), dtype=torch.long)

    cur_beat = 0
    timed_notes: List[Tuple[Fraction, int]] = []
    for t, pitch in song.notes:
        beat = int(t)
        while beat > cur_beat:
            pat = Pattern(tuple(timed_notes))
            patterns[cur_beat] = pattern_map[pat]
            cur_beat += 1
            timed_notes = []
        frac = Fraction(t - beat).limit_denominator(100)
        timed_notes.append((frac, pitch))

    pat = Pattern(tuple(timed_notes))
    patterns[cur_beat] = pattern_map[pat]

    cur_beat = 0
    cur_chord = None
    for t, chord in song.chords:
        beat = round(t)
        while beat > cur_beat:
            if cur_chord is not None:
                if cur_chord in chord_map:
                    chords[cur_beat] = chord_map[cur_chord]
                else:
                    chords[cur_beat] = chord_map[find_replacement_chord(cur_chord)]
            cur_beat += 1

            if cur_beat >= len(chords):
                break
        if cur_beat >= len(chords):
            break

        cur_chord = chord

    while cur_beat < len(chords):
        if cur_chord is not None:
            chords[cur_beat] = chord_map[cur_chord]
        cur_beat += 1

    for i, (beat, number) in enumerate(song.bars[:-1]):
        next_beat, _ = song.bars[i + 1]
        for beat_index in range(int(beat), int(next_beat)):
            if beat_index >= len(bar_numbers):
                continue

            bar_numbers[beat_index] = number
            beat_numbers[beat_index] = beat_index - beat

    ret: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for start in range(0, int(song.duration), config.max_length):
        patterns_patch = patterns[start : start + config.max_length]
        chords_patch = chords[start : start + config.max_length]
        bar_numbers_patch = bar_numbers[start : start + config.max_length]
        beat_numbers_patch = beat_numbers[start : start + config.max_length]

        if len(patterns_patch) < config.max_length:
            padding = (
                torch.ones(config.max_length - len(patterns_patch), dtype=torch.long)
                * config.pad_token_id
            )
            patterns_patch = torch.cat((patterns_patch, padding))
            chords_patch = torch.cat((chords_patch, padding))
            bar_numbers_patch = torch.cat((bar_numbers_patch, padding))
            beat_numbers_patch = torch.cat((beat_numbers_patch, padding))

        ret.append(
            (patterns_patch, chords_patch, bar_numbers_patch, beat_numbers_patch)
        )

    return ret


def find_replacement_chord(chord):
    t = chord.transpose(-chord.bass)
    if t.pitches == (0, 4, 6, 10):
        return Chord(bass=0, root=0, pitches=(0, 4, 10)).transpose(chord.bass)
    if t.pitches == (0, 3, 8, 10):
        return Chord(bass=0, root=8, pitches=(0, 3, 8)).transpose(chord.bass)
    if t.pitches == (0, 4, 6, 11):
        return Chord(bass=0, root=0, pitches=(0, 4, 6, 7, 11)).transpose(chord.bass)
    if t.pitches == (0, 2, 4, 6, 10):
        return Chord(bass=0, root=10, pitches=(0, 2, 6, 10)).transpose(chord.bass)
    raise ValueError("no such chord: " + str(chord))
