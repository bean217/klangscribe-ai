# klangscribe-ai
Repository for KlangScribe Model Training &amp; Inference.

1) [Dataset Preprocessing](#dataset-preprocessing)
2) [Training](#training)
3) [Sparse Note-Event Vocabulary Spec](#guitar-hero-chart-sparse-event-vocabulary) \
    i. [Vocabulary Conventions](#language-conventions)

## Dataset Preprocessing

If using the [KlangScribe Official Dataset](https://drive.google.com/drive/folders/1akCO8kormDrm5N30WHDyIISBr74YWtxT?usp=sharing), the [Dataset Preprocessing](/dataset_preprocessing/README.md) module contains functionality for producing a version of the dataset that KlangScribe can train on directly. Please refer to [it's documentation](/dataset_preprocessing/README.md) for more information on how this is done.

This should be used if you are looking to retrain KlangScribe using a different data modeling configuration. By default, KlangScribe uses the data model configuration template defined [here](/configs/templates/data_model.yaml).


## Training

To train KlangScribe, instead of using an existing model version, following these steps:

1) Preprocess your dataset by following the steps in [Dataset Preprocessing](#dataset-preprocessing)

2) TBD


## Guitar Hero Chart Sparse Event Vocabulary

In order to properly tokenize guitar hero charts, we need to define a method of translating chart files into a learnable language.

The vocabulary we use for KlangScribe is a sparse note-event vocabulary inspired by [MT3: Multi-Task Multitrack Music Transcription](https://arxiv.org/pdf/2111.03017):

#### (1) **Special Tokens** (4 total)
* **[BOS]** - Indicates beginning of a sequence/window
* **[EOS]** - Indicates end of a sequence/window
* **[PAD]** - Pad tokens added after **[EOS]** to fill context
    * Only used during training

#### (2) **Lane On/Off Tokens** (12 total)
* **[Onset(i)]** - Turns a lane on
    * 6x; One for each lane (R, G, Y, B, O, open)
* **[Offset(i)]** - Turns a lane off
    * 6x; One for each lane (R, G, Y, B, O, open)

#### (3) **Onset Modifier Tokens** (2 total)
* **[NoteMod(HOPO)]** - Indicates that an onset is a Hammer-On/Pull-Off
* **[NoteMod(Tap)]** - Indicates that an onset is a Tap
* NOTICE: If no **[NoteMod]** token is given, then it is assumed the note is regular (strummed)

#### (4) **Tie Tokens** (6 total)
* **[Tie(i)]** - Indicates that a sustain continues from the previous window
    * 6x; One for each lane (R, G, Y, B, O, open)

#### (5) **Time-Shift Tokens** ( `ceil[window_len_sec / timestep_len_sec]` )
* **[Time(t)]** - Indicates a jump in time relative to the start of the window
* Example:
    * If `window_len_sec` = 2.0s, `timestep_len_sec` = 0.02s
    * Then `ceil[2.0s / 0.02s]` = 100 time-shift tokens (one for each timestep)

---

### Language Conventions

**General Structure:** tie(i)? ( time(t) ( onsets(i) notemod(m)? offsets(i)? | offsets(i)) )

* The first token is an optional `tie(i)` token for each lane
    * This is only placed if a sustain is being held for this lane frome the previous window
* The second token is always a `time(t)` token, indicating when the event happens
* The third token can be either `onsets(i)` or `offsets(i)`
    * If `onsets(i)`, then:
        * The fourth token is an optional `notemod(m)` token
        * The fifth token is optional `offsets(i)`, indicating a set of lanes **are not sustained**
        * If no fourth nor fifth token is given, then the set of lanes **are sustained**
    * If `offsets(i)`, then this marks the end of a lane's sustain

**Token Type Layouts:**

* `tie(i)`, `onset(i)`, and `offset(i)` tokens are always enumerated in the order (0=G, 1=R, 2=Y, 3=B, 4=O, 5=open)
    * Examples:
        * GRY chord = `[onset(0) onset(1) onset(2)]`
        * RBO chord = `[onset(1) onset(3) onset(4)]`
        * etc.

**Note-Event Examples:**

* Sustained GYO chord with HOPO onset, held for .6s on G, .6s on Y, 1.0s on O
    * `[time(0) onset(0) onset(2) onset(4) notemod(hopo) time(30) offset(0) time(40) offset(2) time(50) offset(4)]`
* End of sustained GRB chord from previous window held for 0.5 seconds into current window
    * `[tie(0) tie(1) tie(3) time(25) offset(0) offset(1) offset(3)]`
* Strummed GY onset (no sustain)
    * `[time(0) onset(0) onset(1) offset(0) offset(1)]`

<!-- REMOVE THE FOLLOWING SECTION:
we don't need to use the SEP token since TIE tokens handle cross-window continuity anyways -->

<!-- **Use of [SEP] for Window Processing:**

During training, our implementation processes 50%-overlapped windows of note events. For example, this means that for a given 2.0s note-event window provided as input into the decoder, all tokens taking place up to 1.0s are followed by a **[SEP]** token:

```bash
(note-event token sequence between t=0s and t=1s)  [SEP]  (not-event token sequence between t=1s and t=2s)
```

During inference, this is useful because the model learns that all input preceding **[SEP]** represents context for autoregressively generating note-events in the second half of the current window. So the decoder for a given input window would be:

```bash
[BOS]  (tokens btwn t=[0,1]s)  [SEP]
```

And the very first window of a song, which has no history, would be:

```bash
[BOS]  [SEP]
``` -->