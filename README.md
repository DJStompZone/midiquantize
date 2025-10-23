# MidiQuantize

Snap a MIDI file to a musical grid and (optionally) lock tempo to 102 BPM.


## Description

Because sloppy timing is charming until it isn't. This snaps note on/off (and optionally pedal) to the nearest grid,
keeps everything else intact, and either forces a single tempo (default: 102 BPM) or preserves the original tempo map.
Grid math is done in ticks so it behaves consistently regardless of the source tempo(s).

## Usage

```sh
python -m midiquantize input.mid -o output.mid --bpm 102 --grid 1/16
python -m midiquantize input.mid -o output.mid --grid 1/8T --strength 0.7 --preserve-tempo
python -m midiquantize input.mid -o output.mid --grid 1/16 --quantize-pedal
```

### Grid options
    - Straight: 1/4, 1/8, 1/16, 1/32
    - Triplets: 1/8T, 1/16T (T = triplet; i.e., 3 notes per beat subdivision)

### Strength
    - 1.0 = full snap to the grid.
    - 0.0 = no movement. Values in between blend toward the target grid.

## Notes
    - Quantizes only note-on/note-off and optionally CC64 (sustain). Other messages keep their absolute positions.
    - When not preserving tempo, the script wipes prior tempo meta-events and inserts a single 102 BPM (or chosen) at t=0.
    - Minimal duration enforcement ensures no zero/negative lengths after snapping.