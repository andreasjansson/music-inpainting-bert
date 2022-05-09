from typing import Dict
import torch
import pretty_midi as pm
import music21 as m21
import numpy as np

from datatypes import Chord, Pattern, Song
import chords
from device import device
from config import HuggingFaceConfig
import data
from preprocess_wikifonia import process_m21_part


def get_predict_mask(mask, mask_prob, num_masked):
    mask_len = int(round(np.ceil(mask_prob * num_masked)))

    # vals, _ = torch.max(pred, axis=1)
    vals = torch.LongTensor(np.random.choice(len(mask), len(mask), False))
    vals[mask] = 0
    vals_sort = torch.argsort(-vals)
    vals_sort_sample = vals_sort[:mask_len]
    predict_mask = (
        torch.nn.functional.one_hot(vals_sort_sample, len(vals)).sum(0).to(bool)
    )
    return predict_mask


def generate_iteration(
    net,
    pattern_vec,
    chord_vec,
    bar_numbers,
    beat_numbers,
    pattern_mask,
    pattern_num_masked,
    chord_mask,
    chord_num_masked,
    config,
    mask_prob,
    iteration_fraction,
    n_patterns_sample,
    n_chords_sample,
    note_onsets,
    chord_onsets,
    pattern_map_inv,
):
    input_patterns = pattern_vec.clone()
    input_chords = chord_vec.clone()

    if pattern_num_masked > 0:
        input_patterns[~pattern_mask] = config.mask_token_id
    if chord_num_masked:
        input_chords[~chord_mask] = config.mask_token_id

    patterns_pred, chords_pred = net(
        input_patterns.reshape([1, -1]),
        input_chords.reshape([1, -1]),
        bar_numbers.reshape([1, -1]),
        beat_numbers.reshape([1, -1]),
    )

    patterns_pred = patterns_pred[0].detach().numpy()
    chords_pred = chords_pred[0].detach().numpy()

    pattern_predict_mask = get_predict_mask(pattern_mask, mask_prob, pattern_num_masked)
    new_pattern_mask = pattern_mask | pattern_predict_mask

    chord_predict_mask = get_predict_mask(chord_mask, mask_prob, chord_num_masked)
    new_chord_mask = chord_mask | chord_predict_mask

    if pattern_num_masked > 0:
        new_pattern_vec = sample_values(
            iteration_fraction,
            n_patterns_sample,
            patterns_pred,
            pattern_predict_mask,
            note_onsets=note_onsets,
            chord_onsets=None,
            pattern_map_inv=pattern_map_inv,
        )
        new_pattern_vec[~new_pattern_mask] = config.pad_token_id
        new_pattern_vec[pattern_mask] = pattern_vec[pattern_mask]
    else:
        new_pattern_vec = pattern_vec

    if chord_num_masked > 0:
        new_chord_vec = sample_values(
            iteration_fraction,
            n_chords_sample,
            chords_pred,
            chord_predict_mask,
            note_onsets=None,
            chord_onsets=chord_onsets,
            pattern_map_inv=pattern_map_inv,
        )
        new_chord_vec[~new_chord_mask] = config.pad_token_id
        new_chord_vec[chord_mask] = chord_vec[chord_mask]
    else:
        new_chord_vec = chord_vec

    return new_pattern_vec, new_chord_vec, new_pattern_mask, new_chord_mask


def sample_values(
    iteration_fraction,
    n_sample,
    values,
    mask,
    note_onsets,
    chord_onsets,
    pattern_map_inv,
):
    assert note_onsets is None or chord_onsets is None

    decay = 5
    n = (
        (n_sample - 1)
        - int((iteration_fraction ** (1 / (n_sample / decay))) * (n_sample - 1))
        + 1
    )
    # n = n - int((iteration_fraction ** (1 / 5)) * n) + 1

    print(iteration_fraction, n)

    output = torch.LongTensor(len(values))

    for i, row in enumerate(values):
        row[:2] = -100
        sort = (-row).argsort()

        if mask[i] == 0:
            continue

        if note_onsets is not None:
            sampled = None
            for index in sort:
                if index < 3:
                    continue

                pat = pattern_map_inv[index]
                onsets = tuple([t for t, _ in pat.timed_notes])
                if onsets != note_onsets[i]:
                    continue

                if sampled is None:
                    sampled = index

                if np.random.randint(n) == 0:
                    sampled = index
                    break

            print(i, sampled, pattern_map_inv[sampled], note_onsets[i])

        else:
            row[:2] = -100
            sort = (-row).argsort()
            row[sort[n:]] = row[sort[n]]

            row = row - np.min(row)
            row = row ** 2
            row = row / np.sum(row)
            sampled = int(np.random.choice(values.shape[1], 1, p=row)[0])

        output[i] = int(sampled)

    return output


def generate_iterations(
    net,
    pattern_vec,
    chord_vec,
    bar_numbers,
    beat_numbers,
    pattern_mask,
    pattern_num_masked,
    chord_mask,
    chord_num_masked,
    config,
    mask_prob,
    n_patterns_sample,
    n_chords_sample,
    note_onsets,
    chord_onsets,
    pattern_map_inv,
):
    net.eval()

    n_iter = int(np.ceil(1 / mask_prob))

    note_iter = [None] * len(pattern_vec)

    for i in range(n_iter):
        for j, m in enumerate(pattern_mask):
            if m and note_iter[j] is None:
                note_iter[j] = i
        pattern_vec, chord_vec, pattern_mask, chord_mask = generate_iteration(
            net=net,
            pattern_vec=pattern_vec,
            chord_vec=chord_vec,
            bar_numbers=bar_numbers,
            beat_numbers=beat_numbers,
            pattern_mask=pattern_mask,
            pattern_num_masked=pattern_num_masked,
            chord_mask=chord_mask,
            chord_num_masked=chord_num_masked,
            config=config,
            mask_prob=mask_prob,
            iteration_fraction=(i / n_iter),
            n_patterns_sample=n_patterns_sample,
            n_chords_sample=n_chords_sample,
            note_onsets=note_onsets,
            chord_onsets=chord_onsets,
            pattern_map_inv=pattern_map_inv,
        )

    for j, m in enumerate(pattern_mask):
        if m and note_iter[j] is None:
            note_iter[j] = n_iter

    return pattern_vec, chord_vec, note_iter


def generate_sequential(net, patterns, chords, mask, config, mask_prob):
    net.train(True)

    patterns[~mask] = config.mask_token_id

    out = torch.zeros(len(patterns), dtype=torch.long)

    for i in range(len(patterns)):
        if patterns[i] == config.mask_token_id:
            pred, _ = net(patterns.reshape([1, -1]), chords.reshape([1, -1]))
            pred = pred[0].argmax(-1).cpu()
            out[i] = pred[i]
        else:
            out[i] = patterns[i]

    return out


def get_chord_vec(bar_chords, time_sig, chord_map, chord_names):
    num_bars = len(bar_chords)
    num_beats = num_bars * time_sig

    chord_vec = torch.zeros(size=[num_beats], dtype=torch.long)
    for i, c in enumerate(bar_chords):
        chord = chord_names[c]
        chord_vec[i * time_sig : i + 1 * time_sig] = chord_map[chord]

    return chord_vec


def generate_masked_song(
    song,
    net,
    chord_map: Dict[Chord, int],
    chord_map_inv: Dict[int, Chord],
    pattern_map: Dict[Pattern, int],
    pattern_map_inv: Dict[int, Pattern],
    chord_names_inv: Dict[Chord, str],
    config: HuggingFaceConfig,
    mask_start=0,
    mask_end=-1,
    tempo: int = 120,
    time_sig: int = 4,
    mask_prob=0.1,
    n_patterns_sample=50,
    n_chords_sample=10,
    max_length=None,
    predict_patterns=True,
    predict_chords=False,
    use_note_onsets=False,
):
    old_max_length = config.max_length
    try:
        config.max_length = 1024
        patches = data.song_to_arrays(song, chord_map, pattern_map)
    finally:
        config.max_length = old_max_length

    pattern_vec, chord_vec, bar_numbers, beat_numbers = patches[0]
    duration = int(song.duration)
    if max_length is not None:
        duration = min(duration, max_length)

    pattern_vec = pattern_vec[:duration]
    chord_vec = chord_vec[:duration]
    bar_numbers = bar_numbers[:duration]
    beat_numbers = beat_numbers[:duration]

    note_onsets = None
    chord_onsets = None
    if use_note_onsets:
        note_onsets = [
            tuple(t for t, _ in pattern_map_inv[int(p)].timed_notes)
            if p > 2
            else tuple()
            for p in pattern_vec
        ]

    if predict_chords:
        chord_vec[mask_start:mask_end] = config.mask_token_id
    if predict_patterns:
        pattern_vec[mask_start:mask_end] = config.mask_token_id

    mask = torch.zeros(len(pattern_vec)).to(bool)
    mask[:] = True
    mask[mask_start:mask_end] = False

    pred_patterns, pred_chords, note_iter = generate_iterations(
        net=net,
        pattern_vec=pattern_vec,
        chord_vec=chord_vec,
        bar_numbers=bar_numbers,
        beat_numbers=beat_numbers,
        pattern_mask=mask,
        pattern_num_masked=mask_end - mask_start,
        chord_mask=mask,
        chord_num_masked=mask_end - mask_start,
        config=config,
        mask_prob=mask_prob,
        n_patterns_sample=n_patterns_sample,
        n_chords_sample=n_chords_sample,
        note_onsets=note_onsets,
        chord_onsets=chord_onsets,
        pattern_map_inv=pattern_map_inv,
    )
    pred_patterns = pred_patterns[:-1].detach().numpy()
    pred_chords = pred_chords[:-1].detach().numpy()

    for i, (p, c, n) in enumerate(zip(pred_patterns, pred_chords, note_iter)):
        if p < 2:
            print(f"{i}: Invalid pattern: {p}")
        else:
            timed_notes = pattern_map_inv[p].timed_notes
            timed_notes_str = ", ".join(
                [
                    f"({t.numerator}/{t.denominator} {m21.pitch.Pitch(n).name}{m21.pitch.Pitch(n).octave})"
                    for t, n in timed_notes
                ]
            )

            if c > 2:
                if chord_map_inv[c] in chord_names_inv:
                    chord_name = chord_names_inv[chord_map_inv[c]]
                else:
                    chord_name = str(chord_map_inv[c])
            else:
                chord_name = "N/A"

            print(
                f"{(i + 1) // time_sig}:{i % time_sig}", timed_notes_str, chord_name, n
            )

    return generate_song(
        pred_patterns,
        pred_chords,
        pattern_map_inv,
        chord_map_inv,
        tempo,
        time_sig,
    )


def generate_from_strings(
    net,
    tinynotation_notes_str,
    chords_str,
    chord_map: Dict[Chord, int],
    chord_map_inv: Dict[int, Chord],
    pattern_map: Dict[Pattern, int],
    pattern_map_inv: Dict[int, Pattern],
    chord_names_inv: Dict[Chord, str],
    config: HuggingFaceConfig,
    tempo: int = 120,
    time_sig: int = 4,
    mask_prob=0.1,
    n_patterns_sample=50,
    n_chords_sample=10,
):
    notes, note_masks, notes_duration = parse_tinynotation_notes(
        tinynotation_notes_str, time_sig
    )
    chords, chord_masks, chords_duration = parse_chords(chords_str, time_sig)
    if notes_duration != chords_duration:
        raise ValueError(
            f"Notes duration ({notes_duration} beats) is not equal to chords duration ({chords_duration} beats)"
        )

    bars = tuple(
        [
            (offset, i + 1)
            for i, offset in enumerate(range(0, int(notes_duration), time_sig))
        ]
    )
    song = Song(notes=notes, chords=chords, bars=bars, duration=notes_duration, path="")

    old_max_length = config.max_length
    try:
        config.max_length = 1024
        patches = data.song_to_arrays(song, chord_map, pattern_map)
    finally:
        config.max_length = old_max_length

    pattern_vec, chord_vec, bar_numbers, beat_numbers = patches[0]
    duration = int(song.duration)

    pattern_vec = pattern_vec[:duration]
    chord_vec = chord_vec[:duration]
    bar_numbers = bar_numbers[:duration]
    beat_numbers = beat_numbers[:duration]

    chord_mask = torch.zeros(len(pattern_vec)).type(torch.bool)
    chord_mask[:] = True
    pattern_mask = torch.zeros(len(pattern_vec)).type(torch.bool)
    pattern_mask[:] = True
    for (mask_start, mask_end) in chord_masks:
        mask_start, mask_end = int(mask_start), int(mask_end)
        chord_vec[mask_start:mask_end] = config.mask_token_id
        chord_mask[mask_start:mask_end] = False
    for (mask_start, mask_end) in note_masks:
        mask_start, mask_end = int(mask_start), int(mask_end)
        pattern_vec[mask_start:mask_end] = config.mask_token_id
        pattern_mask[mask_start:mask_end] = False

    pred_patterns, pred_chords, note_iter = generate_iterations(
        net=net,
        pattern_vec=pattern_vec,
        chord_vec=chord_vec,
        bar_numbers=bar_numbers,
        beat_numbers=beat_numbers,
        pattern_mask=pattern_mask,
        pattern_num_masked=len(pattern_mask) - torch.sum(pattern_mask).item(),
        chord_mask=chord_mask,
        chord_num_masked=len(chord_mask) - torch.sum(chord_mask).item(),
        config=config,
        mask_prob=mask_prob,
        n_patterns_sample=n_patterns_sample,
        n_chords_sample=n_chords_sample,
        note_onsets=None,
        chord_onsets=None,
        pattern_map_inv=pattern_map_inv,
    )
    pred_patterns = pred_patterns[:-1].detach().numpy()
    pred_chords = pred_chords[:-1].detach().numpy()

    for i, (p, c, n) in enumerate(zip(pred_patterns, pred_chords, note_iter)):
        if p < 2:
            print(f"{i}: Invalid pattern: {p}")
        else:
            timed_notes = pattern_map_inv[p].timed_notes
            timed_notes_str = ", ".join(
                [
                    f"({t.numerator}/{t.denominator} {m21.pitch.Pitch(n).name}{m21.pitch.Pitch(n).octave})"
                    for t, n in timed_notes
                ]
            )

            if c > 2:
                if chord_map_inv[c] in chord_names_inv:
                    chord_name = chord_names_inv[chord_map_inv[c]]
                else:
                    chord_name = str(chord_map_inv[c])
            else:
                chord_name = "N/A"

            print(
                f"{(i + 1) // time_sig}:{i % time_sig}", timed_notes_str, chord_name, n
            )

    return generate_song(
        pred_patterns,
        pred_chords,
        pattern_map_inv,
        chord_map_inv,
        tempo,
        time_sig,
    )


def generate_song(
    pattern_vec, chord_vec, pattern_map_inv, chord_map_inv, tempo, time_sig
):
    sec_per_beat = 60 / tempo

    track = pm.PrettyMIDI(initial_tempo=tempo)
    track.time_signature_changes.append(pm.TimeSignature(time_sig, 4, 0))

    lead = pm.Instrument(pm.instrument_name_to_program("Lead 2 (sawtooth)"))
    accomp = pm.Instrument(pm.instrument_name_to_program("Synth Choir"))
    drum = pm.Instrument(0, is_drum=True)

    write_notes(lead, pattern_vec, pattern_map_inv, sec_per_beat)
    track.instruments.append(lead)

    write_chords(accomp, chord_vec, chord_map_inv, sec_per_beat)
    track.instruments.append(accomp)

    write_drum(drum, len(pattern_vec), sec_per_beat)
    track.instruments.append(drum)

    return track


def write_chords(instrument, chord_vec, chord_map_inv, sec_per_beat):
    def append_chord(chord, start_beat, end_beat):
        velocity = 100
        start = start_beat * sec_per_beat
        end = end_beat * sec_per_beat
        note = pm.Note(
            pitch=chord.bass + 12 * 4, velocity=velocity, start=start, end=end
        )
        instrument.notes.append(note)
        for pitch in chord.pitches:
            note = pm.Note(
                pitch=pitch + 12 * 5,
                velocity=velocity,
                start=start,
                end=end,
            )
            instrument.notes.append(note)

    cur = None
    start_beat = 0
    for beat, chord_id in enumerate(chord_vec):
        if chord_id > 1:
            chord = chord_map_inv[int(chord_id.item())]
            if cur is None:
                start_beat = beat
        else:
            chord = None
            print("0 chord")

        if cur is not None and cur != chord:
            append_chord(cur, start_beat, beat)
            start_beat = beat

        cur = chord

    if cur is not None:
        append_chord(cur, start_beat, len(chord_vec))


def write_notes(instrument, pattern_vec, pattern_map_inv, sec_per_beat):
    def append_note(pitch, start_beat, end_beat):
        note = pm.Note(
            pitch=pitch,
            velocity=120,
            start=start_beat * sec_per_beat,
            end=end_beat * sec_per_beat,
        )
        instrument.notes.append(note)

    start_beat = 0
    cur = 0
    for beat, pattern_id in enumerate(pattern_vec):
        if pattern_id > 1:
            pattern = pattern_map_inv[pattern_id]
        else:
            print("%d pattern" % pattern_id)

        for t, p in pattern.timed_notes:
            note_beat = beat + t
            if cur > 0:
                append_note(cur, start_beat, note_beat)
            start_beat = note_beat
            cur = p

    if cur > 0:
        append_note(cur, start_beat, len(pattern_vec))


def write_drum(drum, num_beats, sec_per_beat):
    for i in range(0, num_beats):
        pattern = pm.Note(
            pitch=pm.drum_name_to_note_number("Closed Hi Hat"),
            velocity=100,
            start=i * sec_per_beat,
            end=(i + 0.5) * sec_per_beat,
        )
        drum.notes.append(pattern)


class HarmonyModifier(m21.tinyNotation.Modifier):
    def postParse(self, n):
        cs = m21.harmony.ChordSymbol(n.pitch.name + self.modifierData)
        cs.duration = n.duration
        return cs


def tinynotation_to_song(notes_str: str, chords_str: str):
    tnc = m21.tinyNotation.Converter()
    tnc.load(notes_str)
    notes_part = tnc.parse().stream

    notes, _, bars, duration = process_m21_part(notes_part)

    tnc = m21.tinyNotation.Converter()
    tnc.modifierUnderscore = HarmonyModifier
    tnc.load(chords_str)
    chords_part = tnc.parse().stream

    _, chords, bars_, _ = process_m21_part(chords_part)

    assert bars_ == bars

    return Song(
        notes=tuple(notes),
        chords=tuple(chords),
        bars=tuple(bars),
        duration=duration,
        path="",
    )


def parse_tinynotation_notes(notes_str: str, time_sig: int):
    notes_str = notes_str.strip().lstrip("|").rstrip("|")

    bars_str = notes_str.split("|")
    offset = 0
    all_notes = []
    masks = []

    for i, bar_str in enumerate(bars_str):
        bar_str = bar_str.strip()
        if bar_str == "?":
            duration = time_sig
            masks.append((offset, offset + duration))
        else:
            s = f"{time_sig}/4 {bar_str}"
            tnc = m21.tinyNotation.Converter()
            tnc.load(s)
            notes_part = tnc.parse().stream
            notes, _, bars, duration = process_m21_part(notes_part)

            if duration != time_sig:
                raise AssertionError(
                    f"Bar {i + 1} contains {duration} beats, expected {time_sig}"
                )

            if len(bars) != 1:
                raise AssertionError(f"Bar {i + 1} contains more than one bar of music")

            all_notes += [(t + offset, pitch) for t, pitch in notes]

        offset += duration

    total_duration = offset
    return tuple(all_notes), masks, total_duration


def parse_chords(chords_str: str, time_sig: int):
    chords_str = chords_str.strip().lstrip("|").rstrip("|")

    bars_str = chords_str.split("|")
    offset = 0.0
    all_chords = []
    masks = []

    for i, bar_str in enumerate(bars_str):
        bar_str = bar_str.strip()
        if " " in bar_str:
            raise ValueError(
                f"Error in chord bar {i}, only one chord per bar is allowed"
            )
        if not bar_str:
            raise ValueError(
                f"Error in chord bar {i}, you must provide a chord for each bar"
            )

        if bar_str == "?":
            masks.append((offset, offset + time_sig))
        else:
            try:
                m21_chord = m21.harmony.ChordSymbol(bar_str)
            except Exception:
                raise ValueError(
                    f"Error in chord bar {i}, failed to parse chord: {bar_str}"
                )
            try:
                chord = Chord.from_m21_chord(m21_chord)
            except Exception:
                raise ValueError(f"Error in chord bar {i}, unknown chord: {bar_str}")

            all_chords.append((offset, chord))

        offset += time_sig

    total_duration = offset
    return tuple(all_chords), masks, total_duration


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_maps():
    pattern_map = data.load_pattern_map()
    chord_map = data.load_chord_map()
    chord_names = chords.chords_by_name()

    pattern_map_inv = invert_dict(pattern_map)
    chord_map_inv = invert_dict(chord_map)
    chord_names_inv = invert_dict(chord_names)

    return (
        pattern_map,
        chord_map,
        chord_names,
        pattern_map_inv,
        chord_map_inv,
        chord_names_inv,
    )
