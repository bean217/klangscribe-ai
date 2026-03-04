# KlangScribe AI Dataset Preprocessing

This module contains functionality for constructing a KlangScribe-specific dataset from a canonical form.

## Canonical Dataset Structure

The following canonical dataset structure is expected

```
canonical/
├── song_metadata.parquet
├── raw/
│   ├── audio/
│   │   └── sid_*.opus
│   └── charts/
│       └── sid_*.chart
└── processed/
    ├── instr_audio/
    │   └── sid_*.opus
    ├── vocal_audio/
    │   └── sid_*.opus
    └── charts/
        └── sid_*.npz
```

**Data Splits:**
* `train` (~80%)
* `val` (~10%)
* `test` (~10%)

NOTE: data splits may not adhere to these exact percentages due to issues with managing data leakage

**Data Descriptions:**
* `charts`
    * Contains `.npz` files which hold data pertaining to .chart file structure
    * `resolution` = quarter-note tick resolution
    * `offset` = chart audio time offset (in seconds)
    * `note_data` = note onset events
        * shape=(n, 14) array (n=num ticks)
        * `note_data[:, 0]` = integer ticks (int)
        * `note_data[:, 1:7]` = multi-hot note onsets (g, r, y, b, o, open)
        * `note_data[:, 7:13]` = integer note sustains
        * `note_data[:, 13]` = categorical note type (0 = Regular, 1 = HOPO, 2 = Tap)
    * `tempo_changes` = bpm shift events
        * shape=(n, 2) array (n=num ticks)
        * `tempo_changes[:, 0]` = integer ticks (int)
        * `tempo_changes[:, 1]` = integer BPM value * 1000
            * ex: 120 BPM is represented as `120000`
* `full_songs`
    * Contains `.opus` files which hold full song audio (including vocals)
* `instr_songs`
    * Contains `.opus` files which hold instrumental song audio (excluding vocals)


