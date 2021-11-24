from typing import Dict
import re
import numpy as np
import music21 as m21
from chord_labels import ChordLabel, parse_chord

from datatypes import Chord


def chords_by_name() -> Dict[str, Chord]:
    roots = [
        "C",
        "C#",
        "Db",
        "D",
        "D#",
        "Eb",
        "E",
        "F",
        "F#",
        "Gb",
        "G",
        "G#",
        "Ab",
        "A",
        "A#",
        "Bb",
        "B",
    ]
    qualities = [
        "",
        "m",
        "7",
        "m7",
        "maj7",
        "9",
        "m9",
        "sus4",
        "7sus4",
        "add9",
        "7+5",
        "7-5",
        "m7+5",
        "m7-5",
        "6",
        "m6",
        "7-9",
        "madd9",
        "add9",
        "9+5",
        "13+5",
        "13",
        "11",
        "dim",
    ]

    chord_names = {}
    for root in roots:
        for quality in qualities:
            name = root + quality
            m21_quality = quality.replace("+", "#")
            m21_quality = m21_quality.replace("-", "b")
            m21_root = root.replace("b", "-")
            m21_name = m21_root + m21_quality
            m21_chord = m21.harmony.ChordSymbol(m21_name)
            pitches = tuple(sorted(p.midi % 12 for p in m21_chord.pitches))
            chord_names[name] = Chord(
                pitches=pitches,
                root=m21_chord.root().midi % 12,
                bass=m21_chord.bass().midi % 12,
            )
            if quality in ["", "m"]:
                for bass in roots:
                    name_with_bass = name + "/" + bass
                    chord_names[name_with_bass] = Chord(
                        pitches=pitches,
                        root=m21_chord.root().midi % 12,
                        bass=m21.note.Note(bass).pitch.midi % 12,
                    )


    return chord_names


# based on https://github.com/MTG/JAAH/blob/3690ce748f2b15f04588e2c5ab3940af86582884/utils/jazzomat2json.py
harte_type_mapping = {
    "j79#": "maj7(#9)",
    "79#11#": "7(#9, #11)",
    "+7911#": "maj7(9,11)",
    "j7": "maj7",
    "79b13": "9(b13)",
    "6911#": "maj6(9, #11)",
    "7913": "9(13)",
    "69": "maj6(9)",
    "7911": "9(11)",
    "-79b": "min(b9)",
    "j79#11#": "maj7(#9, #11)",
    "-69": "min6(9)",
    "+": "aug",
    "-": "min",
    "-j7": "minmaj7",
    "sus": "sus4",
    "7": "7",
    "6": "maj6",
    "o7": "dim7",
    "79#": "7(#9)",
    "79#13": "9(#13)",
    "+j7": "(3,#5,7)",
    "7913b": "9(b13)",
    "sus79": "sus4(b7,9)",
    "+79b": "(3,#5,b7, b9)",
    "7alt": "7(b5)",
    "sus7": "sus4(b7)",
    "j7911#": "maj7(9, #11)",
    "+79": "aug(7, 9)",
    "79": "9",
    "sus7913": "sus4(b7, 9, 13)",
    "j79": "maj7(9)",
    "m7b5": "hdim7",
    "-7913": "min7(9, 13)",
    "-79": "min9",
    "-j7913": "minmaj7(9,13)",
    "o": "dim",
    "-7": "min7",
    "-6": "min6",
    "+7": "aug(7)",
    "79b": "7(b9)",
    "+79#": "aug(7, #9)",
    "-j7911#": "min7(9, #11)",
    "79b13b": "7(b9, b13)",
    "7911#": "9(#11)",
    "-7911": "min7(9, 11)",
}

degrees = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]


def degree(root, bass):
    root_n = pitch.Pitch(root).midi % 12
    bass_n = pitch.Pitch(bass).midi % 12
    degree = bass_n - root_n
    if degree < 0:
        degree += 12
    return degrees[degree]


cache = {}
harte_cache = {}


def harte(chord: str) -> ChordLabel:
    if chord in cache:
        return cache[chord]

    if chord == "NC" or chord == "":
        return parse("N")
    else:
        r = re.search("^([ABCDEFG][b#]?)([^/]*)(/([ABCDEFG][b#]?))?$", chord)
        root = r.group(1)
        type = r.group(2)
        bass = r.group(4)
        res = [root]
        if type != "":
            res.append(":")
            res.append(harte_type_mapping[type])
        if bass != None:
            res.append("/")
            res.append(degree(root, bass))
    harte_chord = parse("".join(res))
    cache[chord] = harte_chord
    return harte_chord


def parse(title: str) -> ChordLabel:
    if title in harte_cache:
        return harte_cache[title]
    chord = parse_chord(title)
    harte_cache[title] = chord
    return chord


transpose_re = re.compile(r"^([^:/]+)(?::([^/]+))?(?:/(.+))?$")


transpose_cache = {}


def _transpose(c: ChordLabel, step: int) -> ChordLabel:
    title = c.title
    if (title, step) in transpose_cache:
        return transpose_cache[(title, step)]

    r = transpose_re.search(title)
    groups = r.groups()
    root = groups[0]
    quality = groups[1]
    bass = groups[2]

    if root == "N":
        transpose_cache[(title, step)] = c
        return c

    transposed_root = pitch.Pitch(root).transpose(step).name.replace("-", "b")
    parts = [transposed_root]
    if quality is not None:
        parts.append(":" + quality)
    if bass is not None:
        parts.append("/" + bass)

    ret = parse_chord("".join(parts))
    transpose_cache[(title, step)] = ret
    return ret


ChordLabel.transpose = _transpose
