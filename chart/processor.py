#
# file: chart/processor.py
# desc: Provides functionality for processing raw .chart data into a vectorized format
# auth: Benjamin Piro (brp8396@rit.edu)
# date: 2026-02-28
#


import numpy as np


def _process_tempo_changes(tempo_changes: np.ndarray, resolution: int):
    """
    Converts tempo changes from BPM to absolute time in seconds based on the specified resolution.
    
    Args:
    - tempo_changes (np.ndarray): An array of shape (num_tempo_changes, 2) where each row contains [tick, bpm] for a tempo change event.
        - tick (int): the tick at which the tempo change occurs
        - bpm (int): the tempo in beats-per-minute mutliplied by 1000 (e.g., 120 BPM would be represented as 120000)
    - resolution (int): The resolution of the chart in ticks per quarter note, used to calculate the timing
    """
    segments = []
    last_tick = 0
    cum_time = 0.0

    for tick, bpm in tempo_changes:
        if tick < last_tick:
            raise ValueError(f"Tempo changes must be sorted by tick in ascending order. Found tick {tick} after tick {last_tick}.")
        
        # normalize BPM
        bpm_norm = bpm / 1000.0

        # calculate time elapsed since last tempo change in seconds
        if segments:
            prev_bpm = segments[-1][1]
            delta_ticks = tick - last_tick
            delta_time = (delta_ticks / resolution) * (60.0 / prev_bpm)
            cum_time += delta_time
        
        segments.append((tick, bpm_norm, cum_time))
        last_tick = tick
    
    # store as a structured numpy array
    dtypes = [('tick', int), ('bpm', float), ('time_sec', float)]
    return np.array(segments, dtype=dtypes)


def _tick_to_sec(tick: int, bpm_ticks: np.ndarray, resolution: int) -> float:
    """
    Converts a tick value to absolute time in seconds based on the provided BPM changes and resolution.
    
    Args:
        - tick (int): The tick position to convert to seconds.
        - bpm_ticks: A structured numpy array containing the BPM changes with fields 'tick', 'bpm', and 'time_sec'.
        - resolution (int): The resolution of the chart in ticks per quarter note. 
    """

    if tick < 0:
        raise ValueError(f"Tick value cannot be negative. Received tick: {tick}")
    if bpm_ticks is None or bpm_ticks.shape[0] == 0:
        raise ValueError('BPM ticks cannot be None or empty.')
    if resolution <= 0:
        raise ValueError(f"Resolution must be a positive integer. Received resolution: {resolution}")
    
    # Find the appropriate BPM segment for the given tick using binary search
    idx = np.searchsorted(bpm_ticks['tick'], tick, side='right') - 1

    # Handle edge case where tick is before the first BPM change
    if idx < 0:
        # if tick is before first bpm event, use first bpm
        idx = 0
        base_time = 0.0
        bpm = bpm_ticks['bpm'][0]
        delta_ticks = tick
    else:
        # otherwise use the bpm and time from the found segment
        base_tick, bpm, base_time = bpm_ticks[idx]
        delta_ticks = tick - base_tick

    delta_time = (delta_ticks / resolution) * (60.0 / bpm)
    return base_time + delta_time


######################
#                    #
#   Static Methods   #
#                    #
######################

@staticmethod
def convert_to_abstime(note_data: np.ndarray, tempo_changes, resolution: int, offset: float = 0.0):
    """
    Converts vectorized chart data into its absolute-time representation.
    
    Args:
        - note_data (np.ndarray): The vectorized representation of the chart data to be converted into absolute-time format, which contains information about the timing and state of notes in the chart.
        - tempo_changes: The tempo changes in the chart, which are used to calculate the absolute timing of events based on the resolution and offset.
        - resolution (int): The resolution of the chart, which is used to calculate the timing of events in ticks and convert them to absolute time based on the tempo changes and offset.
        - offset (float): The offset time in seconds to be applied to all events in the chart, which can be used to align the timing of events with audio or other reference points.
    """
    if note_data is None:
        raise ValueError("Note data cannot be None.")
    if tempo_changes is None or tempo_changes.shape[0] == 0:
        raise ValueError("Tempo changes cannot be None or empty.")
    if resolution <= 0:
        raise ValueError("Resolution must be a positive integer.")
    if offset < 0:
        raise ValueError("Offset cannot be negative.")
    
    # convert BPM events to absolute time in sections
    processed_tempo_changes = _process_tempo_changes(tempo_changes, resolution)
    note_times = []

    for note_frame in note_data:
        # retrieve note frame data
        note_tick = note_frame[0]               # assuming the first column contains the tick information
        note_press_data = note_frame[1:7]       # assuming the next 6 columns contain the note press data for each lane
        note_sustain_data = note_frame[7:13]    # assuming the next 6 columns contain the sustain data for each lane
        note_type = note_frame[13]              # assuming the last column contains the note type information (e.g., normal, HOPO, tap, etc.)
        
        # convert note tick to absolute time in seconds
        start_sec = _tick_to_sec(note_tick, processed_tempo_changes, resolution)

        note_press_durations = []
        for i in range(note_press_data.shape[0]):
            # calculate duration o fsustains based on sustain length data and BPM events
            if note_press_data[i] == 1 and note_sustain_data[i] > 0:
                end_tick = note_tick + note_sustain_data[i]
                end_sec = _tick_to_sec(end_tick, processed_tempo_changes, resolution)
                duration_sec = end_sec - start_sec
            else:
                duration_sec = 0.0
            
            # append duration of this note press to the list of note press durations for this note frame
            note_press_durations.append(duration_sec)

        # apply song offset (in seconds) to start_sec to get absolute time for this note frame
        absolute_time_sec = start_sec + offset

        # create new note frame with absolute time and the rest of the note data
        note_times.append([absolute_time_sec] + note_press_data.tolist() + note_press_durations + [note_type])

    # convert list of absolute-time note frames to numpy array
    note_times = np.array(note_times, dtype=float) 
    return note_times    


@staticmethod
def validate_chart(abstime_chart_data: np.ndarray, min_delta_time: int) -> bool:
    """
    Validates the absolute-time chart data against specified criteria, such as minimum delta time between events.
    
    Args:
        - abstime_chart_data (np.ndarray): The absolute-time representation of the chart data to be validated.
        - min_delta_time (int): The minimum allowed delta time between events in the chart data, used for validation.
    Returns:
        - bool: True if the chart data is valid according to the specified criteria, False otherwise
    """
    raise NotImplementedError("Chart validation logic not yet implemented.")

@staticmethod
def convert_to_fixed_grid(abstime_chart_data: np.ndarray, grid_size: int):
    """
    Converts absolute-time chart data into a fixed-grid representation based on the specified grid size.
    
    Args:
        - abstime_chart_data (np.ndarray): The absolute-time representation of the chart data to be converted.
        - grid_size (int): The size of the grid to be used for converting the chart data into a fixed-grid representation.
    """
    raise NotImplementedError("Fixed-grid conversion logic not yet implemented.")

@staticmethod
def convert_to_event_based(fixed_grid_chart_data: np.ndarray):
    """
    Converts fixed-grid chart data into an event-based format, where each event corresponds to a change in the state of the chart (e.g., note on, note off, etc.).
    
    Args:
        - fixed_grid_chart_data (np.ndarray): The fixed-grid representation of the chart data to be converted into event-based format.
    """
    raise NotImplementedError("Event-based conversion logic not yet implemented.")

@staticmethod
def chunk_chart_data(event_based_chart_data: np.ndarray, context_length: int):
    """
    Chunks event-based chart data into overlapping windows based on the specified context length, padding as necessary.
    
    Args:
        - event_based_chart_data (np.ndarray): The event-based representation of the chart data to be chunked into overlapping windows.
        - context_length (int): The length of the context window to be used for chunking the chart data, which determines the shape of the resulting chunked data (num_chunks, context_length, num_features).
    """
    raise NotImplementedError("Chart chunking logic not yet implemented.")


if __name__ == "__main__":
    # Example usage of the chart processing functionality
    from pathlib import Path
    import chart.reader as chart_reader
    sample_chart_path = Path(__file__).parent / "data" / "sample_output.npz"
    resolution, offset, tempo_changes, note_data = chart_reader.read_vectorized_chart(sample_chart_path)
    print("Resolution:", resolution)
    print("Offset:", offset)
    print("Tempo Changes:", tempo_changes)
    print("Note Data:", note_data)

    print("\nConverting to absolute time representation...")
    abs_time_arr = convert_to_abstime(note_data, tempo_changes, resolution, offset)
    for row in abs_time_arr:
        time_sec = float(row[0])
        press = [int(x) for x in row[1:7]]
        sustain = [int(x) for x in row[7:13]]
        note_type = int(row[13])
        print(f"Time: {time_sec:.3f} sec, Press: {press}, Sustain: {sustain}, Note Type: {note_type}")