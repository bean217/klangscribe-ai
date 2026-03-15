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


def _merge_similar_events(unmerged_events_arr):
    """
    Given an array of sorted, unmerged, absolute-time note events, combines note events of the same
    event type, note type, and occurrence time into a single event.

    Input Shape: (n, 8)
    Output Shape: (m, 8) where m <= n, and n-m is the number of merges performed
    """
    if len(unmerged_events_arr) <= 1:
        # no need to try merging
        return unmerged_events_arr
    
    def merge_events(note_events: list, similar_events: list):
        if len(similar_events) == 1:
                note_events.append(similar_events[0])
        else:
            time_sec = similar_events[0][0:1]
            lanes = np.max([evt[1:7] for evt in similar_events], axis=0)
            note_type = similar_events[0][7:8]
            evt_type = similar_events[0][8:9]
            note_events.append(np.hstack((time_sec, lanes, note_type, evt_type), dtype=float))

    # merge events with the same event type and note type that occur at the same time
    note_events = []
    similar_events = unmerged_events_arr[:1]
    for note_evt in unmerged_events_arr[1:]:
        last_time = similar_events[-1][0]
        last_note_type = similar_events[-1][7]
        last_evt_type = similar_events[-1][8]
        cur_time = note_evt[0]
        cur_note_type = note_evt[7]
        cur_evt_type = note_evt[8]
        # if the current event happens at the same time as the last and has the same type, merge
        # "at the same time" = "within one microsecond"
        if np.abs(last_time - cur_time) < 1e-6 and last_note_type == cur_note_type and last_evt_type == cur_evt_type:
            similar_events.append(note_evt)
        else:
            # the current event is different from the previous so merge the events in merge_events
            merge_events(note_events, similar_events)
            # then add the current event to it        
            similar_events = [note_evt]
    # merge any remaining events
    if len(similar_events) > 0:
        merge_events(note_events, similar_events)
    # return the accumulated note events
    return note_events


########################
#                      #
#   Public Functions   #
#                      #
########################


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
            # calculate duration of sustains based on sustain length data and BPM events
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



def validate_chart(abstime_chart_data: np.ndarray, min_delta_time: int) -> bool:
    """
    Validates the absolute-time chart data against specified criteria, such as minimum delta time between events.
    
    Args:
        - abstime_chart_data (np.ndarray): The absolute-time representation of the chart data to be validated.
        - min_delta_time (int): The minimum allowed delta time between events in the chart data, used for validation.
    Returns:
        - bool: True if the chart data is valid according to the specified criteria, False otherwise
    """
    
    delta_times = np.diff(abstime_chart_data[:,0])
    for dt in delta_times:
        print("Delta Time:", dt)

    raise NotImplementedError("Chart validation logic not yet implemented.")


def convert_to_fixed_grid(abstime_event_chart_data: np.ndarray, grid_size: float):
    """
    Converts absolute-time chart data into a fixed-grid representation based on the specified grid size.
    
    Args:
        - abstime_chart_data (np.ndarray): The absolute-time representation of the chart data to be converted.
        - grid_size (float): The time width of the grid to be used for converting the chart data into a fixed-grid representation.
    """
    # calculate quantized tick times
    quantized_times = np.round(abstime_event_chart_data[:,0] / grid_size)
    # transplant these into a copy of the original data
    quantized_chart_data = abstime_event_chart_data.copy()
    quantized_chart_data[:,0] = quantized_times
    # return int-casted quantized data
    return quantized_chart_data.astype(int)


def convert_to_event_based(abstime_chart_data: np.ndarray):
    """
    Converts absolute-time chart data into an event-based format, where each event corresponds to a change in the state of the chart (e.g., note on, note off, etc.).
    
    Args:
        - abstime_chart_data (np.ndarray): The absolute-time representation of the chart data to be converted into event-based format.
    
    Returns:
        - numpy matrix with:
            - cols[0] = event time (float; seconds)
            - cols[1:7] = affected lanes (binary)
            - cols[7] = note type (0=regular, 1=HOPO, 2=Tap; ignored if event type is 0)
            - cols[8] = note event type (binary; 0=offset, 1=o########nset)
    """
    # use a priority queue (heapq) to keep events sorted
    note_events = []
    for chart_data in abstime_chart_data:
        # (1) extract the onset event info
        time_sec = chart_data[0:1]
        onset_lanes = chart_data[1:7]
        onset_type = chart_data[13:14]
        note_evt_type = 1
        # stack into one row vector
        onset_evt_arr = np.hstack((time_sec, onset_lanes, onset_type, note_evt_type), dtype=float)
        note_events.append(onset_evt_arr)

        # (2) extract the offset event info
        if any(sustain_lanes := chart_data[7:13]):
            unique_sustain_lengths = np.unique(sustain_lanes)
            # construct offset event for the end of each unique sustain length
            for sus_len in unique_sustain_lengths:
                if not sus_len:
                    # skip zero-length sustains
                    continue
                # calculate offset event time marker
                time_sec = chart_data[0:1] + sus_len
                # determine affected lanes
                lane_inds = np.where(sustain_lanes == sus_len)[0]
                offset_lanes = np.zeros_like(sustain_lanes)
                offset_lanes[lane_inds] = 1
                note_evt_type = offset_type = 0
                # stack into one row vector
                offset_evt_arr = np.hstack((time_sec, offset_lanes, offset_type, note_evt_type), dtype=float)
                note_events.append(offset_evt_arr)

    # sort events first by the time they occur, then by event type
    note_events.sort(key=lambda note_evt: (note_evt[0], -note_evt[8]))
    # merge note events which occur at the same time and are of the same type
    note_events = _merge_similar_events(note_events)
    # stack vertically and return
    event_chart_data = np.vstack(note_events)
    return event_chart_data


def chunk_chart_data(event_based_chart_data: np.ndarray, context_length: int, overlap_length: int = 0):
    """
    Chunks event-based chart data into overlapping windows based on the specified context length, padding as necessary.
    
    Args:
        - event_based_chart_data (np.ndarray): The event-based representation of the chart data to be chunked into overlapping windows.
        - context_length (int): The tick length of the context window to be used for chunking the chart data, which determines the shape of the resulting chunked data (num_chunks, context_length, num_features).
        - overlap_length (int): The number of ticks by which consecutive windows should overlap. Default is 0 (no overlap).
    """
    if overlap_length < 0 or overlap_length >= context_length:
        raise ValueError(f"Overlap length must be non-negative and less than context length. Received overlap_length: {overlap_length}, context_length: {context_length}")
    if event_based_chart_data is None or event_based_chart_data.shape[0] == 0:
        return []   # song has no events
    event_based_chart_data = event_based_chart_data.copy()
    cur_win_tick = 0
    windows = []
    # construct windows by sliding a window of size context_length across the chart data with a step size of (context_length - overlap_length)
    while cur_win_tick <= event_based_chart_data[-1][0]:
        mask = (cur_win_tick <= event_based_chart_data[:,0]) & (event_based_chart_data[:,0] < cur_win_tick + context_length)
        window_events = event_based_chart_data[mask]
        if window_events.shape[0] == 0:
            windows.append([])
        else:
            # event times are relative to the start of the window
            window_events[:,0] -= cur_win_tick
            windows.append(window_events)
        cur_win_tick += context_length - overlap_length
    # remove the last window if it contains no new events (i.e., if the last window has ticks that are all less than the overlap length)
    if overlap_length > 0 and len(windows) >= 2 and windows[-1][:,0].max() < overlap_length:
        windows.pop()
    return windows


if __name__ == "__main__":
    # Example usage of the chart processing functionality
    from pathlib import Path
    import chart.reader as chart_reader
    sample_chart_path = Path(__file__).parent / "data" / "reddi_theshow.npz"
    resolution, offset, tempo_changes, note_data = chart_reader.read_vectorized_chart(sample_chart_path)
    print("Resolution:", resolution)
    print("Offset:", offset)
    print("Tempo Changes:", tempo_changes)
    print("Note Data:", note_data)

    print("\nConverting to absolute time representation...")
    abs_time_arr = convert_to_abstime(note_data, tempo_changes, resolution, offset)
    # for row in abs_time_arr:
    #     time_sec = float(row[0])
    #     press = [int(x) for x in row[1:7]]
    #     sustain = [float(x) for x in row[7:13]]
    #     note_type = int(row[13])
    #     print(f"Time: {time_sec:.3f} sec, Press: {press}, Sustain: {sustain}, Note Type: {note_type}")

    # validate_chart(abs_time_arr, 0.02)

    print("\nConverting to abstime event representation...")
    evt_chart_data = convert_to_event_based(abs_time_arr)
    for row in evt_chart_data:
        time_sec = float(row[0])
        lanes = [int(x) for x in row[1:7]]
        note_type = int(row[7])
        evt_type = "Onset" if row[8] else "Offset"
        print(f"Time: {time_sec:.3f} sec, Lanes: {lanes}, Note Type: {note_type}, Event Type: {evt_type}")
    
    print("\nQuantizing abstime event chart data using grid_size = 0.02 (20ms)")
    quantized_evt_chart_data = convert_to_fixed_grid(evt_chart_data, grid_size=0.02)
    for row in quantized_evt_chart_data:
        time_sec = int(row[0])
        lanes = [int(x) for x in row[1:7]]
        note_type = int(row[7])
        evt_type = "Onset" if row[8] else "Offset"
        print(f"Tick: {time_sec}, Lanes: {lanes}, Note Type: {note_type}, Event Type: {evt_type}")

    print("\nChunking chart events into windows with context_len=100...")
    chunked_chart_data = chunk_chart_data(quantized_evt_chart_data, context_length=100, overlap_length=50)
    for i, window in enumerate(chunked_chart_data):
        print("Window", i)
        if len(window) == 0:
            print("\tEmpty")
            continue
        for row in window:
            time_sec = int(row[0])
            lanes = [int(x) for x in row[1:7]]
            note_type = int(row[7])
            evt_type = "Onset" if row[8] else "Offset"
            print(f"\tTick: {time_sec}, Lanes: {lanes}, Note Type: {note_type}, Event Type: {evt_type}")
