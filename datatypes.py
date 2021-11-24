# need this class because autoreloading frozen dataclasses is broken
# in jupyter https://github.com/ipython/ipython/issues/12185

from fractions import Fraction
from typing import List, Tuple
from dataclasses import dataclass
import music21 as m21


@dataclass(frozen=True, eq=True)
class Chord:
    pitches: Tuple[int, ...]
    root: int
    bass: int

    def __str__(self):
        pitches_str = ",".join([str(p) for p in self.pitches])
        return f"{pitches_str}/{self.root}/{self.bass}"

    @classmethod
    def from_string(cls, s):
        pitches_str, root, bass = s.split("/")
        pitches = tuple(int(x) for x in pitches_str.split(","))
        root = int(root)
        bass = int(bass)
        return Chord(pitches, root, bass)

    @classmethod
    def from_m21_chord(cls, m21_chord: m21.harmony.ChordSymbol):
        pitches = tuple(sorted(set([int(n.pitch.midi) % 12 for n in m21_chord.notes])))
        root = int(m21_chord.root().midi % 12)
        bass = int(m21_chord.bass().midi % 12)
        return Chord(pitches, root, bass)

    @classmethod
    def from_dict(cls, d: dict):
        return Chord(
            pitches=tuple(sorted(d["pitches"])), root=d["root"], bass=d["bass"],
        )

    def to_dict(self):
        return {
            "pitches": self.pitches,
            "root": self.root,
            "bass": self.bass,
        }

    def transpose(self, step):
        return Chord(
            tuple(sorted((p + step) % 12 for p in self.pitches)),
            (self.root + step) % 12,
            (self.bass + step) % 12,
        )


@dataclass(frozen=True, eq=True)
class Song:
    notes: Tuple[Tuple[float, int], ...]
    chords: Tuple[Tuple[float, Chord], ...]
    bars: Tuple[Tuple[float, int], ...]
    duration: float
    path: str

    @classmethod
    def from_dict(cls, d: dict, path: str):
        return Song(
            notes=tuple((t, n) for t, n in d["notes"]),
            chords=tuple((t, Chord.from_dict(c)) for t, c in d["chords"]),
            bars=tuple((t, n) for t, n in d["bars"]),
            duration=d["duration"],
            path=path,
        )

    def transpose(self, step):
        notes = tuple((t, n + step) for t, n in self.notes)
        chords = tuple((t, c.transpose(step)) for t, c in self.chords)
        return Song(
            notes=notes,
            chords=chords,
            bars=self.bars,
            duration=self.duration,
            path=self.path,
        )


@dataclass(frozen=True, eq=True)
class Pattern:
    timed_notes: Tuple[Tuple[Fraction, int], ...]

    @classmethod
    def from_list(cls, lst):
        timed_notes = tuple((Fraction(frac[0], frac[1]), pitch) for frac, pitch in lst)
        return Pattern(timed_notes=timed_notes)
