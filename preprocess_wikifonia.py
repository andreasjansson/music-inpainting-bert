import dataclasses
import json
from glob import glob
import os
from multiprocessing import Pool
from typing import Tuple, List
import music21 as m21

from datatypes import Chord


def process(path):
    out_name = "preprocessed-json/" + os.path.basename(path) + ".json"
    if os.path.exists(out_name):
        return

    print(path)
    try:
        notes, chords, bars, duration = load_wikifonia_score(path)
    except Exception as e:
        print("Error processing %s: %s" % (path, e))
        return

    out_name = "preprocessed-json/" + os.path.basename(path) + ".json"
    try:
        with open(out_name, "w") as f:
            json.dump(
                {
                    "notes": notes,
                    "chords": [(t, dataclasses.asdict(c)) for t, c in chords],
                    "bars": bars,
                    "duration": duration,
                },
                f,
            )
    except Exception as e:
        try:
            os.unlink(out_name)
        except:
            pass
        print("Failed to save %s: %s" % (out_name, e))


def load_wikifonia_score(
    path: str,
) -> Tuple[
    List[Tuple[float, int]], List[Tuple[float, Chord]], List[Tuple[float, int]], float
]:
    s = m21.converter.parse(path)
    if len(s.parts) != 1:
        print(f"Wikifonia {path} has {len(s.parts)} parts")

    part = s.parts[0]

    return process_m21_part(part)


def process_m21_part(
    part,
) -> Tuple[
    List[Tuple[float, int]], List[Tuple[float, Chord]], List[Tuple[float, int]], float
]:
    notes: List[Tuple[float, int]] = []
    chords: List[Tuple[float, Chord]] = []
    bars: List[Tuple[float, int]] = []

    measures = part.getElementsByClass(m21.stream.Measure)
    duration = 0

    for measure in measures:
        bars.append((measure.offset, measure.number))

        offsets = measure.offsetMap()
        for offset in offsets:
            t = float(offset.offset + measure.offset)
            el = offset.element
            if isinstance(el, m21.harmony.ChordSymbol):
                chords.append((t, Chord.from_m21_chord(el)))
            elif isinstance(el, m21.note.Note):
                ties_prev = (
                    el.tie is not None
                    and el.tie.type == "stop"
                    and el.pitch.midi == notes[-1][1]
                )

                if len(notes) > 0:
                    if notes[-1][0] >= t:
                        raise Exception("Notes are not sorted")

                if not ties_prev and not el.duration.isGrace:
                    notes.append((t, el.pitch.midi))
                    duration = t + offset.endTime - offset.offset

    return notes, chords, bars, duration


def main():
    paths = list(sorted(glob("Wikifonia/*.mxl")))
    with Pool(32) as p:
        p.map(process, paths)


if __name__ == "__main__":
    main()
