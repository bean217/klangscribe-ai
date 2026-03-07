#
# file: chart/writer.py
# desc: Provides functionality for constructing .chart files from alternative data formats,
#       such as vectorized or tokenized representations of chart data.
# auth: Benjamin Piro (brp8396@rit.edu)
# date: 2026-02-28
#

import numpy as np
import torch as th
from string import Template
from chart.vocab import ChartVocab


SONG_SECTION_TEMPLATE = """
[Song]
{
Resolution = $resolution
Offset = $offset
}
"""

SYNCT_RACK_SECTION_TEMPLATE = """
[SyncTrack]
{
0 = TS 4
0 = B $bpm
}
"""

NOTE_SECTION_TEMPLATE = """
[ExpertSingle]
{
$note_events
}
"""

NOTE_EVENT_TEMPLATE = "$tick = N $event $duration"


######################
#                    #
#   Public Methods   #
#                    #
######################


def write_chart_from_vec(output_path: str, quantized_note_data: np.ndarray, bpm: float, resolution: int):
    """
    Constructs a .chart file from quantized vectorized chart data and writes it to the specified output path.
    Assumes that the quantized note data is aligned to a fixed grid based on the specified resolution and BPM.
    """
    note_section = Template(SONG_SECTION_TEMPLATE).substitute(offset=0.0, resolution=resolution)
    print(note_section)


def write_chart_from_tokens(output_path: str, vocab: ChartVocab, token_sequence: list[int], bpm: float, resolution: int):
    """
    Constructs a .chart file from a sequence of token IDs representing the chart data and writes it to the specified output path.
    Assumes that the token sequence is a sequence of token IDs that can be mapped back to quantized note data based on the specified resolution and BPM.
    """
    raise NotImplementedError("Chart writing logic not yet implemented.")


if __name__ == "__main__":
    # Example usage of the chart writing functionality
    output_path = "output_chart.chart"
    resolution = 192
    bpm = 120.0

    write_chart_from_vec(output_path=output_path, quantized_note_data=np.array([]), bpm=bpm, resolution=resolution)