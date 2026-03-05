import pytest
import numpy as np
from pathlib import Path
from chart.reader import parse_chart, save_vectorized_chart, read_vectorized_chart

SAMPLE_FILE = Path(__file__).parent / "data" / "sample_input.chart"

def test_parse_chart():
    resolution, offset, tempo_changes, note_events = parse_chart(SAMPLE_FILE)
    
    assert isinstance(resolution, int)
    assert resolution == 192, f"Expected resolution to be 192 ticks per quarter note, got {resolution}."

    assert isinstance(offset, float)
    assert offset == 0.0, f"Expected offset to be 0.0 seconds, got {offset}."

    assert isinstance(tempo_changes, np.ndarray)
    assert tempo_changes.shape == (28, 2), f"Expected tempo changes to have shape (28, 2), got {tempo_changes.shape}."

    assert isinstance(note_events, np.ndarray)
    assert note_events.shape == (493, 14), f"Expected note events to have shape (493, 14), got {note_events.shape}."