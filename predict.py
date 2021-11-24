# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import subprocess
import os
import tempfile
from pathlib import Path

import cog
import torch
import numpy as np
from midiSynth.synth import MidiSynth

import model
import config
import generate


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.net = model.LeadSheetForMaskedLM.from_pretrained("pretrained")
        self.config = config.HuggingFaceConfig()
        (
            self.pattern_map,
            self.chord_map,
            self.chord_names,
            self.pattern_map_inv,
            self.chord_map_inv,
            self.chord_names_inv,
        ) = generate.load_maps()

        self.midi_synth = MidiSynth()

    @cog.input(
        "notes",
        type=str,
        help="Notes in tinynotation, with each bar separated by '|'. Use '?' for bars you want in-painted.",
    )
    @cog.input(
        "chords",
        type=str,
        help="Chords (one chord per bar), with each bar separated by '|'. Use '?' for bars you want in-painted.",
    )
    @cog.input(
        "time_signature",
        type=int,
        options=[3, 4, 5, 7],
        default=4,
        help="Time signature",
    )
    @cog.input(
        "tempo", type=int, min=60, max=200, default=120, help="Tempo (beats per minute)"
    )
    @cog.input(
        "sample_width",
        type=int,
        min=1,
        max=200,
        default=10,
        help="Number of potential predictions to sample from. The higher, the more chaotic the output.",
    )
    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    @cog.input(
        "output_format",
        type=str,
        options=["mp3", "midi"],
        default="mp3",
        help="Output file format (synthesized mp3 audio or raw midi)",
    )
    def predict(self, notes, chords, time_signature, tempo, sample_width, seed, output_format):
        """Run a single prediction on the model"""

        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        out_dir = Path(tempfile.mkdtemp())
        midi_path = out_dir / "out.midi"
        wav_path = out_dir / "out.wav"
        mp3_path = out_dir / "out.mp3"

        generate.generate_from_strings(
            net=self.net,
            tinynotation_notes_str=notes,
            chords_str=chords,
            chord_map=self.chord_map,
            chord_map_inv=self.chord_map_inv,
            pattern_map=self.pattern_map,
            pattern_map_inv=self.pattern_map_inv,
            chord_names_inv=self.chord_names_inv,
            config=self.config,
            filename=str(midi_path),
            n_patterns_sample=sample_width,
            time_sig=time_signature,
            tempo=tempo,
        )
        if output_format == "midi":
            return midi_path

        try:
            self.midi_synth.midi2audio(str(midi_path), str(wav_path))
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-i",
                    str(wav_path),
                    "-af",
                    "silenceremove=1:0:-50dB,aformat=dblp,areverse,silenceremove=1:0:-50dB,aformat=dblp,areverse",  # strip silence
                    str(mp3_path),
                ],
            )
            return mp3_path
        finally:
            midi_path.unlink()
            wav_path.unlink(missing_ok=True)
