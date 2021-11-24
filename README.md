# Joint melody and chord inpainting with BERT

<a href="https://replicate.ai/andreasjansson/music-inpainting-bert"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=darkgreen" height=20></a>

## Model

This project uses a custom BERT model that is masking both melody and chords in the same piece of music.

## Data representation

The model takes as input beat-quantized chord labels and beat-quantized _melodic patterns_.

### Melodic patterns

Melodies are split into beat-sized chunks, where each chunk is quantized to 16th notes. The chunks are stored in a look-up table.

## Dataset

The model is trained on the Wikifonia dataset.
