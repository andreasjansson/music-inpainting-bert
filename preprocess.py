import math
from collections import Counter
from fractions import Fraction
from datatypes import Chord
import json


def write_pattern_map(songs):
    frac_cache = {}

    patterns = Counter()
    for i, song in enumerate(songs):
        if i % 100 == 0:
            print(i)
        for step in range(12):
            s = song.transpose(step)
            cur_pat = []
            cur_beat = 0
            for t, pitch in sorted(s.notes):
                if int(t) > cur_beat:
                    patterns[tuple(cur_pat)] += 1
                    cur_pat = []
                    cur_beat = int(t)
                start = t - cur_beat
                if start in frac_cache:
                    time_frac = frac_cache[start]
                else:
                    time_frac = Fraction((t - cur_beat)).limit_denominator(100)
                    frac_cache[start] = time_frac
                cur_pat.append(((time_frac.numerator, time_frac.denominator), pitch))

            patterns[tuple(cur_pat)] += 1

    pattern_map = {(i + 2): pat for i, (pat, _) in enumerate(patterns.most_common())}
    with open("pattern_map.json", "w") as f:
        json.dump(pattern_map, f)

    return pattern_map


def write_chord_map(songs):
    chords = Counter()
    for i, song in enumerate(songs):
        if i % 100 == 0:
            print(i)
        for step in range(12):
            s = song.transpose(step)
            for _, chord in sorted(s.chords, key=lambda x: x[0]):
                chords[chord] += 1

    chord_map = {
        (i + 2): chord.to_dict() for i, (chord, _) in enumerate(chords.most_common())
    }
    with open("chord_map.json", "w") as f:
        json.dump(chord_map, f)

    return chord_map
