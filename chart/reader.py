#
# file: chart/reader.py
# desc: Provides functionality for reading .chart files and converting them into a basic vectorized format.
# auth: Benjamin Piro (brp8396@rit.edu)
# date: 2026-03-04
#

import re
import numpy as np
import logging
import torch as th
from enum import Enum
from abc import ABC, abstractmethod
from utils.logging import format_exception


logger = logging.getLogger(__name__)


class ChartSection(ABC):
    def __init__(self, name: str):
        self.name = name


# -------------------------- #
#   [Song] Section Parsing   #
# -------------------------- #


class SongMetadataSection(ChartSection):
    def __init__(self, name: str):
        super().__init__(name)
        # Resolution is the only required field in the [Song] section
        self.resolution: int = None
        self.offset: float = 0.0
    
    def append(self, key: str, value: str):
        match key:
            case "Resolution":
                self.resolution = int(value)
            case "Offset":
                # This value may be in an international format, with a comma as the decimal separator
                self.offset = float(value.replace(',', '.'))
            case _:
                # We can ignore any other fields in the [Song] section; they aren't needed
                pass


METADATA_REGEX = r'(Resolution|Offset)\s*=\s*"?([^"\n]+)"?'


def _parse_song_section(section_content: str) -> SongMetadataSection:
    """Parses content of the [Song] section in a .chart file."""
    song_metadata = SongMetadataSection(name="Song")
    for line in section_content.splitlines():
        match = re.match(METADATA_REGEX, line.strip())
        if match:
            key, value = match.group(1), match.group(2)
            song_metadata.append(key, value)
    return song_metadata


# ----------------------------- #
#   SyncTrack Section Parsing   #
# ----------------------------- #


class TempoChangeEvent:
    def __init__(self, tick: int, bpm: float):
        self.tick = tick    # the tick at which the tempo change occurs
        self.bpm = bpm      # the BPM value for this tempo change, multiplied by 1000
    
    def to_numpy(self) -> np.ndarray:
        # convert this tempo change marker to a numpy array representation
        # structure: [tick, bpm]
        return np.array([self.tick, self.bpm], dtype=int)


class SyncTrackSection(ChartSection):
    def __init__(self, name: str):
        super().__init__(name)
        self.tempo_changes: list[TempoChangeEvent] = []

    def append(self, tick: int, bpm: int):
        evt = TempoChangeEvent(tick=tick, bpm=bpm)
        self.tempo_changes.append(evt)
    
    def to_numpy(self) -> np.ndarray:
        # convert the list of tempo change events to a numpy array representation
        return np.array([evt.to_numpy() for evt in self.tempo_changes], dtype=int)


SYNCTRACK_REGEX = r'(\d+)\s*=\s*B\s*(\d+)'  # matches lines like "0 = B 120000" (tick 0, BPM 120.000)


def _parse_synctrack_section(section_content: str) -> SyncTrackSection:
    """Parses content of the [SyncTrack] section in a .chart file."""
    synctrack_data = SyncTrackSection(name="SyncTrack")
    for line in section_content.splitlines():
        match = re.match(SYNCTRACK_REGEX, line.strip())
        if match:
            tick, bpm = int(match.group(1)), int(match.group(2))
            synctrack_data.append(tick, bpm)
    return synctrack_data


# ------------------------ #
#   Note Section Parsing   #
# ------------------------ #


class NoteType(Enum):
    REGULAR = 0
    HOPO = 1
    TAP = 2


# Clone Hero .chart file ID mapping for frets
fret_mapping = {
    0: 'green',
    1: 'red',
    2: 'yellow',
    3: 'blue',
    4: 'orange',
    7: 'open'
}


class Fret:
    pressed: bool = False
    sustain_length: int = 0


class NoteEvent:
    def __init__(self, tick: int):
        self.tick = tick
        self.frets = {
            'green': Fret(),    # corresponds to the green fret lane in Clone Hero
            'red': Fret(),      # corresponds to the red fret lane in Clone Hero
            'yellow': Fret(),   # corresponds to the yellow fret lane in Clone Hero
            'blue': Fret(),     # corresponds to the blue fret lane in Clone Hero
            'orange': Fret(),   # corresponds to the orange fret lane in Clone Hero
            'open': Fret(),     # corresponds to open notes in Clone Hero, which require no fret to be pressed
                                #   but may be combined with other frets for chords
        }
        self.note_type = NoteType.REGULAR  # default to regular note (strummed); may be updated to HOPO or TAP based on note modifiers in the .chart file

    def update(self, fret_id: int, sustain_length: int):
        # add a fret to this note event
        if (0 <= fret_id <= 4) or fret_id == 7:
            fret_name = fret_mapping[fret_id]
            self.frets[fret_name].pressed = True
            self.frets[fret_name].sustain_length = sustain_length
        elif fret_id == 5:
            self.note_type = NoteType.HOPO
        elif fret_id == 6:
            self.note_type = NoteType.TAP
        else:
            raise ValueError(f"Invalid fret ID: {fret_id} in note event at tick {self.tick}")
    
    def to_numpy(self) -> np.ndarray:
        # convert this note event to a numpy array representation
        # structure: [
        #   tick,
        #   green_pressed, red_pressed, yellow_pressed, blue_pressed, orange_pressed, open_pressed,
        #   green_sustain, red_sustain, yellow_sustain, blue_sustain, orange_sustain, open_sustain,
        #   note_type
        # ]

        # (1) add tick that the note occurs at
        fret_data = [self.tick]

        # (2) add fret pressed data and sustain length data for each fret
        fret_names = ['green', 'red', 'yellow', 'blue', 'orange', 'open']
        for fret_name in fret_names:
            fret_data.append(int(self.frets[fret_name].pressed))
        for fret_name in fret_names:
            fret_data.append(self.frets[fret_name].sustain_length)
        
        # (3) add note type data
        fret_data.append(self.note_type.value)

        return np.array(fret_data, dtype=int)


class InstrumentTrackSection(ChartSection):
    """
    Represents a single instrument track section in a .chart file, which contains note events
    for a specific instrument (e.g., ExpertSingle guitar track)
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.note_events: list[NoteEvent] = []
    
    def append(self, tick: int, chart_val: int, sustain_length: int):
        is_new_note = (not self.note_events) or (tick != self.note_events[-1].tick)
        marker = NoteEvent(tick=tick) if is_new_note else self.note_events[-1]
        try:
            marker.update(chart_val, sustain_length)
            if is_new_note:
                self.note_events.append(marker)
        except Exception as e:
            # When Clone Hero encounters malformed note data in a .chart file, it ignores it
            logger.warning(f"Skipping note at tick {tick} with invalid chart value {chart_val} due to error: {format_exception(e)}")

    def to_numpy(self) -> np.ndarray:
        # convert the note events in this instrument track section to a numpy array representation
        return np.array([evt.to_numpy() for evt in self.note_events], dtype=int)


NOTE_REGEX = r'(\d+)\s*=\s*N\s*(\d+)\s*(\d+)'  # matches lines like "480 = N 0 120" (note at tick 480, green fret, sustain length 120)


def _parse_notes_section(section_name: str, section_content: str) -> InstrumentTrackSection:
    """Parses content of an instrument track section in a .chart file."""
    instrument_track = InstrumentTrackSection(name=section_name)  # for now, we can hardcode the name since we'll only be parsing the ExpertSingle guitar track
    for line in section_content.splitlines():
        match = re.match(NOTE_REGEX, line.strip())
        if match:
            tick, chart_val, sustain_length = int(match.group(1)), int(match.group(2)), int(match.group(3))
            instrument_track.append(tick, chart_val, sustain_length)
    return instrument_track


###########################
#                         #
#   Chart Parsing Logic   #
#                         #
###########################


SECTION_REGEXES = {
    section_header: re.compile(rf'\[{section_header}\]\s*\{{(.*?)\}}', re.DOTALL)
    for section_header in ("Song", "SyncTrack", "ExpertSingle",)
}


def parse_chart(chart_path: str) -> tuple[int, float, np.ndarray, np.ndarray]:
    """
    Reads a .chart file from the specified path and converts it into a vectorized format, saving the result to the output path.
    """
    # ensure that the input file is a .chart file
    assert str(chart_path).endswith('.chart'), "Input file must be a .chart file containing raw chart data."

    # extracted chart section data
    section_content = {}

    # read .chart file content into a buffer
    with open(chart_path, 'r') as f:
        chart_data = f.read()
    
    # extract sections using regex
    for section, regex in SECTION_REGEXES.items():
        match = regex.search(chart_data)
        if match:
            section_content[section] = match.group(1)
        else:
            logger.warning(f"Section [{section}] not found in chart file {chart_path}. This may indicate a malformed .chart file or one that doesn't conform to expected structure.")
            raise ValueError(f"Section [{section}] not found in chart file {chart_path}")
    
    # extract relevant data from each section
    song_metadata = None
    if "Song" in section_content:
        song_metadata = _parse_song_section(section_content["Song"])
    synctrack_data = None
    if "SyncTrack" in section_content:
        synctrack_data = _parse_synctrack_section(section_content["SyncTrack"])
    expert_single_data = None
    if "ExpertSingle" in section_content:
        expert_single_data = _parse_notes_section("ExpertSingle", section_content["ExpertSingle"])
    
    # ensure that we successfully extracted the necessary sections and data
    assert all(section_data is not None for section_data in [song_metadata, synctrack_data, expert_single_data]), \
        "Failed to extract necessary sections from .chart file; cannot proceed with processing."

    # return song resolution, offset, tempo changes, and note events as numpy arrays
    return (
        song_metadata.resolution,
        song_metadata.offset,
        synctrack_data.to_numpy(),
        expert_single_data.to_numpy()
    )


def save_vectorized_chart(resolution: int, offset: float, tempo_changes: np.ndarray, note_events: np.ndarray, output_path: str):
    """
    Saves the vectorized chart data to the specified output path in .npz format.
    """
    np.savez(output_path, resolution=resolution, offset=offset, tempo_changes=tempo_changes, note_events=note_events)


def read_vectorized_chart(vectorized_chart_path: str) -> tuple[int, float, np.ndarray, np.ndarray]:
    """
    Reads vectorized chart data from the specified .npz file path and returns it as a tuple containing resolution, offset, tempo changes, and note events.
    """
    # ensure that the input file is a .npz file containing vectorized chart data
    assert str(vectorized_chart_path).endswith('.npz'), "Input file must be a .npz file containing vectorized chart data."
    
    with np.load(vectorized_chart_path) as data:
        resolution = data['resolution']
        offset = data['offset']
        tempo_changes = data['tempo_changes']
        note_events = data['note_events']
    return resolution, offset, tempo_changes, note_events


if __name__ == "__main__":
    # example usage
    from pathlib import Path
    chart_path = Path(__file__).parent / "data" / "sample_input.chart"
    output_path = Path(__file__).parent / "data" / "sample_output.npz"
    resolution, offset, tempo_changes, note_events = parse_chart(chart_path)

    print("Note events:", note_events.shape)
    for row in note_events:
        print([int(v) for v in row])
    print("Tempo changes:", tempo_changes.shape)
    for row in tempo_changes:
        print([int(v) for v in row])
    print("Resolution:", resolution)
    print("Offset:", offset)

    save_vectorized_chart(resolution, offset, tempo_changes, note_events, output_path)