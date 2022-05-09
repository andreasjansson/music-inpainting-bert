# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import subprocess
import os
import tempfile

from cog import BaseModel, BasePredictor, Input, Path
import torch
import numpy as np
from midiSynth.synth import MidiSynth

import model
import config
import generate


class Output(BaseModel):
    mp3: Path
    score: Path
    midi: Path


class Predictor(BasePredictor):
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

    def predict(
        self,
        notes: str = Input(
            description="Notes in tinynotation, with each bar separated by '|'. Use '?' for bars you want in-painted."
        ),
        chords: str = Input(
            description="Chords (one chord per bar), with each bar separated by '|'. Use '?' for bars you want in-painted."
        ),
        time_signature: int = Input(
            default=4, choices=[3, 4, 5, 7], description="Time signature"
        ),
        tempo: int = Input(
            default=120, ge=60, le=200, description="Tempo (beats per minute)"
        ),
        sample_width: int = Input(
            default=10,
            ge=1,
            le=200,
            description="Number of potential predictions to sample from. The higher, the more chaotic the output.",
        ),
        seed: int = Input(default=-1, description="Random seed, -1 for random"),
    ) -> Output:
        """Run a single prediction on the model"""

        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        out_dir = Path(tempfile.mkdtemp())
        midi_path = out_dir / "out.midi"
        midi_path_no_drums = out_dir / "out-no-drums.midi"
        wav_path = out_dir / "out.wav"
        mp3_path = out_dir / "out.mp3"
        lilypond_path = out_dir / "score.ly"
        score_path = out_dir / "score.png"

        # Generate prettymidi file
        track = generate.generate_from_strings(
            net=self.net,
            tinynotation_notes_str=notes,
            chords_str=chords,
            chord_map=self.chord_map,
            chord_map_inv=self.chord_map_inv,
            pattern_map=self.pattern_map,
            pattern_map_inv=self.pattern_map_inv,
            chord_names_inv=self.chord_names_inv,
            config=self.config,
            n_patterns_sample=sample_width,
            time_sig=time_signature,
            tempo=tempo,
        )
        track.write(str(midi_path))

        # Remove drums for score generation
        track.instruments = track.instruments[:2]
        track.write(str(midi_path_no_drums))

        # Generate wav audio from midi
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
        except Exception as e:
            print(f"Failed to save mp3: {e}")
        finally:
            wav_path.unlink(missing_ok=True)

        # Generate sheet music with lilypond from midi
        try:
            subprocess.check_output(
                ["midi2ly", str(midi_path_no_drums), "--output", str(lilypond_path)],
            )
            subprocess.check_output(
                [
                    "lilypond",
                    "-fpng",
                    "-dresolution=300",
                    '-dpaper-size="a5landscape"',
                    "-o",
                    str(score_path.with_suffix("")),
                    str(lilypond_path),
                ]
            )
        except Exception as e:
            print(f"Failed to save score: {e}")
        finally:
            lilypond_path.unlink(missing_ok=True)

        return Output(
            mp3=mp3_path,
            score=score_path,
            midi=midi_path,
        )
