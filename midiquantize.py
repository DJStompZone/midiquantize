#!/usr/bin/env python3
"""
midiquantize.py — Snap a MIDI file to a musical grid and (optionally) lock tempo to 102 BPM.

Author: DJ Stomp <85457381+DJStompZone@users.noreply.github.com>
License: MIT
Repo: https://github.com/djstompzone/midiquantize

Usage:
    python midiquantize.py input.mid -o output.mid --bpm 102 --grid 1/16
    python midiquantize.py input.mid -o output.mid --grid 1/8T --strength 0.7 --preserve-tempo
    python midiquantize.py input.mid -o output.mid --grid 1/16 --quantize-pedal

Why this exists:
    Because sloppy timing is charming until it isn't. This snaps note on/off (and optionally pedal) to the nearest grid,
    keeps everything else intact, and either forces a single tempo (default: 102 BPM) or preserves the original tempo map.
    Grid math is done in ticks so it behaves consistently regardless of the source tempo(s).

Grid options:
    - Straight: 1/4, 1/8, 1/16, 1/32
    - Triplets: 1/8T, 1/16T (T = triplet; i.e., 3 notes per beat subdivision)
Strength:
    - 1.0 = full snap to the grid.
    - 0.0 = no movement. Values in between blend toward the target grid.

Notes:
    - We quantize only note-on/note-off and optionally CC64 (sustain). Other messages keep their absolute positions.
    - When not preserving tempo, the script wipes prior tempo meta-events and inserts a single 102 BPM (or chosen) at t=0.
    - Minimal duration enforcement ensures no zero/negative lengths after snapping.

Docstring tone policy:
    Keep it honest, keep it helpful, and if your timing is a mess — we fix it, we don’t judge it.
"""

from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import mido

# --------------------------- Utilities ---------------------------

def bpm_to_mido_tempo(bpm: float) -> int:
    """
    Convert BPM to the MIDI 'set_tempo' value (microseconds per beat).
    """
    return int(round(60_000_000 / bpm))


def parse_grid(grid: str, ticks_per_beat: int) -> int:
    """
    Translate a grid string (e.g., '1/16', '1/8T') to a tick subdivision size.

    Args:
        grid: One of {'1/4','1/8','1/16','1/32','1/8T','1/16T'}
        ticks_per_beat: TPB from the MIDI file.

    Returns:
        grid_ticks: number of ticks per grid division.

    Raises:
        ValueError: if the grid string is unsupported.
    """
    g = grid.strip().lower()
    triplet = g.endswith("t")
    base = g[:-1] if triplet else g

    if not base.startswith("1/"):
        raise ValueError(f"Unsupported grid format: {grid}")

    try:
        denom = int(base.split("/")[1])
    except Exception as exc:
        raise ValueError(f"Unsupported grid format: {grid}") from exc

    # Straight subdivision: denom notes per whole note, so denom/4 per beat (since a beat = quarter note).
    # Ticks per division = ticks_per_beat / (denom / 4) = 4 * ticks_per_beat / denom
    # Triplets: 3 notes per two straight subdivisions, i.e., multiply straight step by 2/3.
    straight_ticks = (4 * ticks_per_beat) // denom
    if triplet:
        # Using integer math; we keep it precise by multiplying first then integer divide.
        grid_ticks = (straight_ticks * 2) // 3
        if grid_ticks == 0:
            grid_ticks = 1
        return grid_ticks
    else:
        return max(1, straight_ticks)


def blend_quantize(original: int, target: int, strength: float) -> int:
    """
    Blend the original tick time toward the quantized target by 'strength'.

    Args:
        original: original absolute tick
        target: snapped absolute tick
        strength: 0..1 where 1 = full snap

    Returns:
        int tick time after partial quantization.
    """
    if strength >= 1.0:
        return target
    if strength <= 0.0:
        return original
    # Linear interpolation on ticks, rounded to nearest int
    blended = round(original + strength * (target - original))
    return int(blended)


def safe_duration(start_tick: int, end_tick: int, min_ticks: int = 1) -> Tuple[int, int]:
    """
    Ensure note duration is at least 'min_ticks' and end >= start.
    """
    if end_tick <= start_tick:
        end_tick = start_tick + min_ticks
    return start_tick, end_tick


@dataclass
class NoteEvent:
    """
    A paired MIDI note with absolute tick times.
    """
    start_tick: int
    end_tick: int
    channel: int
    note: int
    velocity: int
    track_index: int
    on_index: int
    off_index: int


# --------------------------- Core Logic ---------------------------

def collect_absolute_times(track: mido.MidiTrack) -> List[int]:
    """
    Compute absolute tick times for each message in a track.
    """
    abs_times: List[int] = []
    t = 0
    for msg in track:
        t += msg.time
        abs_times.append(t)
    return abs_times


def index_notes(track: mido.MidiTrack, track_idx: int, abs_times: List[int]) -> List[NoteEvent]:
    """
    Build paired note events from a track by matching note_on and note_off (or note_on vel=0).

    We use a simple per-(channel, note) stack for pairing. Handles overlapping same-note repeats.
    """
    stacks: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    notes: List[NoteEvent] = []

    for i, msg in enumerate(track):
        if msg.type == "note_on" and msg.velocity > 0:
            key = (msg.channel if hasattr(msg, "channel") else 0, msg.note)
            stacks.setdefault(key, []).append((i, abs_times[i], msg.velocity))
        elif msg.type in ("note_off", "note_on"):
            # Treat note_on with velocity 0 as note_off
            if msg.type == "note_on" and msg.velocity != 0:
                continue
            key = (msg.channel if hasattr(msg, "channel") else 0, msg.note)
            if key in stacks and stacks[key]:
                on_index, on_time, vel = stacks[key].pop(0)
                off_index, off_time = i, abs_times[i]
                notes.append(NoteEvent(start_tick=on_time, end_tick=off_time, channel=key[0], note=msg.note, velocity=vel, track_index=track_idx, on_index=on_index, off_index=off_index))
            # If unmatched, we ignore the orphan off; it's safer than forcing bogus pairs.

    return notes


def rebuild_track_with_quantized_notes(track: mido.MidiTrack, abs_times: List[int], notes: List[NoteEvent], grid_ticks: int, strength: float, quantize_pedal: bool) -> mido.MidiTrack:
    """
    Rebuild a track by replacing the original note_on/note_off events at quantized times.
    Other messages keep their original absolute times (except optional CC64 quantization).

    Args:
        track: original MidiTrack
        abs_times: absolute tick times per message index
        notes: paired notes to quantize
        grid_ticks: ticks per quantization division
        strength: 0..1
        quantize_pedal: if True, quantize CC64 sustain events too

    Returns:
        New MidiTrack with quantized notes and preserved ordering (by absolute time).
    """
    # Map original message indices for on/off so we can remove them.
    to_remove = set()
    for n in notes:
        to_remove.add(n.on_index)
        to_remove.add(n.off_index)

    # Gather "non-note" events (or notes we will reinsert after quantize)
    events: List[Tuple[int, mido.Message]] = []
    for i, msg in enumerate(track):
        if i in to_remove:
            continue
        if quantize_pedal and msg.type == "control_change" and getattr(msg, "control", None) == 64:
            # Sustain pedal — quantize its absolute time
            orig = abs_times[i]
            snapped = round(orig / grid_ticks) * grid_ticks
            qt = blend_quantize(orig, snapped, strength)
            events.append((qt, msg.copy(time=0)))
        else:
            events.append((abs_times[i], msg.copy(time=0)))

    # Quantize notes and add them back
    for n in notes:
        q_on_target = round(n.start_tick / grid_ticks) * grid_ticks
        q_off_target = round(n.end_tick / grid_ticks) * grid_ticks
        q_on = blend_quantize(n.start_tick, q_on_target, strength)
        q_off = blend_quantize(n.end_tick, q_off_target, strength)
        q_on, q_off = safe_duration(q_on, q_off, min_ticks=1)

        on_msg = mido.Message("note_on", note=n.note, velocity=n.velocity, channel=n.channel, time=0)
        off_msg = mido.Message("note_off", note=n.note, velocity=0, channel=n.channel, time=0)
        events.append((q_on, on_msg))
        events.append((q_off, off_msg))

    # Sort by absolute time, stable for same-time ordering
    events.sort(key=lambda x: x[0])

    # Rebuild deltas
    new_track = mido.MidiTrack()
    last_abs = 0
    for abs_t, msg in events:
        delta = abs_t - last_abs
        last_abs = abs_t
        msg.time = max(0, delta)
        new_track.append(msg)
    return new_track


def quantize_file(path_in: str, path_out: str, bpm: float, grid: str, strength: float, preserve_tempo: bool, quantize_pedal: bool, log: Optional[logging.Logger] = None) -> None:
    """
    Load a MIDI, quantize note timings to the given grid, optionally rewrite tempo, and save.

    Args:
        path_in: input MIDI path
        path_out: output MIDI path
        bpm: target BPM for tempo rewriting (ignored if preserve_tempo=True)
        grid: string like '1/16' or '1/8T'
        strength: 0..1 partial quantization toward grid
        preserve_tempo: if True, keep original tempo map; else set single tempo at t=0
        quantize_pedal: if True, quantize sustain (CC64) messages too
        log: optional logger
    """
    mid = mido.MidiFile(path_in)
    tpb = mid.ticks_per_beat
    grid_ticks = parse_grid(grid, tpb)
    logger = log or logging.getLogger(__name__)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"File: {path_in}, ticks_per_beat={tpb}, grid='{grid}' => grid_ticks={grid_ticks}, strength={strength}, preserve_tempo={preserve_tempo}, quantize_pedal={quantize_pedal}")

    new_tracks: List[mido.MidiTrack] = []

    # If not preserving tempo, we will strip out set_tempo metas and emit one at t=0 in the first track later.
    for ti, track in enumerate(mid.tracks):
        abs_times = collect_absolute_times(track)
        notes = index_notes(track, ti, abs_times)

        # Build a filtered track copy that will later be merged with quantized notes
        rebuilt = rebuild_track_with_quantized_notes(track, abs_times, notes, grid_ticks, strength, quantize_pedal)

        if not preserve_tempo:
            # Remove set_tempo events from this track; we'll place a single global tempo in track 0.
            filtered = mido.MidiTrack()
            for msg in rebuilt:
                if not (msg.is_meta and msg.type == "set_tempo"):
                    filtered.append(msg)
            new_tracks.append(filtered)
        else:
            new_tracks.append(rebuilt)

    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)

    if not preserve_tempo:
        if not new_tracks:
            new_tracks.append(mido.MidiTrack())
        # Ensure the first track exists and starts with a tempo event at time 0.
        first = new_tracks[0]
        tempo_msg = mido.MetaMessage("set_tempo", tempo=bpm_to_mido_tempo(bpm), time=0)

        # If the very first event already has some delta, we can just insert tempo with time=0; next event keeps its delta.
        first.insert(0, tempo_msg)

        # Optional: could also ensure a time_signature is present, but not required for quantization correctness.

    # Attach tracks to output file
    for t in new_tracks:
        out.tracks.append(t)

    out.save(path_out)


# --------------------------- CLI ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quantize MIDI note timings to a musical grid and (optionally) set a single fixed tempo (default: 102 BPM).")
    p.add_argument("input", help="Input .mid file")
    p.add_argument("-o", "--output", required=True, help="Output .mid file")
    p.add_argument("--bpm", type=float, default=102.0, help="Target BPM when rewriting the tempo map (ignored with --preserve-tempo). Default: 102.0")
    p.add_argument("--grid", default="1/16", choices=["1/4", "1/8", "1/16", "1/32", "1/8T", "1/16T"], help="Quantization grid. Default: 1/16")
    p.add_argument("--strength", type=float, default=1.0, help="Quantization strength 0..1 (1.0 = full snap). Default: 1.0")
    p.add_argument("--preserve-tempo", action="store_true", help="Preserve the original tempo map instead of forcing a single tempo at time 0.")
    p.add_argument("--quantize-pedal", action="store_true", help="Also quantize sustain pedal (CC64) events to the grid.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    logger = logging.getLogger("midiquantize")
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not (0.0 <= args.strength <= 1.0):
        raise SystemExit("strength must be between 0.0 and 1.0")

    quantize_file(path_in=args.input, path_out=args.output, bpm=args.bpm, grid=args.grid, strength=args.strength, preserve_tempo=args.preserve_tempo, quantize_pedal=bool(args.quantize_pedal), log=logger)


if __name__ == "__main__":
    main()
