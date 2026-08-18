<div align="center">
  <h1>Completing Unfinished Bach</h1>
  <h3>Roman analysis on Bach's chorales — with no note information</h3>
</div>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![music21](https://img.shields.io/badge/music21-Corpus%20%26%20Roman%20Analysis-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Metrics-informational)
![Pandas](https://img.shields.io/badge/Pandas-DataFrames-informational)
![Seaborn](https://img.shields.io/badge/Seaborn-Heatmaps-informational)
![FluidSynth](https://img.shields.io/badge/FluidSynth-Audio%20Rendering-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20%7C%20Jupyter-lightgrey)
[![GitHub Repo stars](https://img.shields.io/github/stars/ACM40960/completing-unfinished-bach?style=social)](https://github.com/ACM40960/completing-unfinished-bach/stargazers)

🎼 Have you ever thought that behind a piece of music there are harmonic dependencies with relations potentially stronger than the sound itself? This project models the harmonic relations between Roman numerals and their properties (major, minor, inverted, and so on) in Bach's chorales — **without using notes or MIDI data at all**. From the music21 Bach chorale corpus to a model-ready DataFrame, and on to custom neural networks that complete Bach's unfinished work.

## Table of Contents

- [Abstract](#abstract)
- [Project Description](#project-description)
  - [Why Roman Numerals?](#why-roman-numerals)
  - [Key Components](#key-components)
- [Data](#data)
  - [From music21's corpus to an explorable DataFrame](#from-music21s-corpus-to-an-explorable-dataframe)
  - [Variables](#variables)
- [Models](#models)
  - [Markov baseline](#1-markov-baseline)
  - [Linear model](#2-linear-model)
  - [GRU / LSTM](#3-gru--lstm)
  - [Chained LSTM](#4-chained-lstm-final-model)
- [Results](#results)
- [Hearing the Result](#hearing-the-result)
- [Repository Structure](#repository-structure)
- [Running the Notebooks](#running-the-notebooks)
- [Limitations and Future Work](#limitations-and-future-work)
- [Project Poster](#project-poster)
- [References](#references)
- [Author](#author)

## Abstract

The harmonic conventions of the Baroque exist to let a composer direct expectation through perceptually grounded patterns, expressed as musical rules. This project presents an analysis of Bach's chorales that models the harmonic relations between Roman numerals and their properties (major, minor, inverted, and so on) without using notes or MIDI data at all. A first-order Markov analysis gives a cross-entropy ≈ 2.20, equal to a perplexity ≈ 9 — the number of possible chords drops from 14 possible choices to 9 by looking only at the previous chord. The order of the different methods that were tested in order to conclude to a final model is: **Markov → linear → GRU → LSTM → chained**, trained with 362 chorales and 29,795 chords, split by chorale (262 train / 50 validation / 50 test). The final chained LSTM reaches **48.3% top-1 / 73.0% top-3 accuracy** on the next Roman degree, and out of roughly 300 possible complete chord specifications it reproduced Bach's own choice exactly — degree, quality, inversion, seventh and duration all matching the Riemenschneider harmonisation — in **18.3%** of cases. Translated back into notes, the two versions sound very close (**85%** audio similarity).

## Project Description

### Why Roman Numerals?

The harmonic conventions of the Baroque exist to let a composer direct expectation through perceptually grounded patterns, expressed as musical rules:

| Move | Effect |
|---|---|
| V → I | resolution, closure |
| V → vi | deceptive, suspended |
| IV → V | preparation, curiosity about what follows |

These moves live at the level of *harmonic function*, not individual notes. Working directly in Roman numeral space asks a sharper question than note-level modelling — and shifts the goal from reproducing Bach to **continuing his intentions**.

### Key Components

- **Data extraction:** search all of the files (Bach's chorales) included in the corpus — the ones with 4 voices — and organise everything based on the bar, an important task for Roman numeral analysis, creating a dataset with roman_scale_degrees, chord_quality, inversion, has7 and beat_strength.
- **Markov analysis:** a heat map to inspect connections when we change from one chord (Roman number) to the other, with the probability that the second chord will occur given that the first chord has occurred.
- **Model sweeps:** LSTM, GRU and a linear model trained for each combination of hyperparameters (window size, units, dropout).
- **Chained LSTM:** each time a chord has been predicted, the results from all predictors are added as an observation for the next prediction.
- **Audio rendering:** converting a chorale and its predicted version to .wav — Bach's part first, and after that the prediction.

## Data

### From music21's corpus to an explorable DataFrame

One of the safest and cleanest ways (without any noise) to obtain Bach's chorales into a dataset is through [music21](https://web.mit.edu/music21/) — a well-known, trustworthy library that contains scores for the purpose of being studied, with more than just the notes turned into data (Roman numeral chords, key, anacrusis, offset, etc.). However, the data lives in nested stream structures that need careful processing to become a readable, model-ready table.

**Step 1 — Bach's chorales in music21's corpus.** The corpus is searched with `corpus.search(sourcePath='bach', numberOfParts=4)`, returning the 368 four-voice chorale files (see music21's [User's Guide, Chapter 11: Corpus Searching](https://music21.org/music21docs/usersGuide/usersGuide_11_corpusSearching.html) for the general mechanism). Alternatives such as `corpus.chorales.Iterator()` or downloading scores online were examined and rejected — the first only limits what `corpus.search` already provides, the second introduces noise while the corpus already contains all of Bach's chorales accurately.

**Step 2 — Duplicates and different harmonic interpretations.** Chorales sharing a BWV id can be duplicate formats of the same piece (.krn vs .mxl) or genuinely different harmonic interpretations of the same hymn. BWV ids are extracted from the file paths; .krn duplicates are removed (the same chorale already exists in .mxl form), and one chorale present in two versions is dropped entirely to prevent any data leak. Chorales containing non-Roman chords (French/German/Italian augmented sixths) are also removed whole — deleting a single chord would disturb the natural flow of the progression.

**Step 3 — A bar-aware, offset-indexed table.** Analysis of a Baroque piece starts by recognising the key (`analyze('key')`). Measure data is extracted via `getElementsByClass('Measure')`, mapping music21's attributes `number`, `numberSuffix`, `offset`, `duration.quarterLength`, `paddingLeft` to the columns `bar`, `suffix`, `offset`, `len`, `anacrusis`. One subtlety matters: **bar numbers are not unique** — when the suffix is not empty, the bar number repeats (e.g. bar `4` and bar `4a`), so the offset, not the bar number, is used to traverse a chorale.

<details>
<summary>Example — navigating bwv101.7 per bar and per SATB voice (click to expand)</summary>

```
current bar: 0
Soprano
{0.0} <music21.key.Key of d minor>
{0.0} <music21.meter.TimeSignature 4/4>
{0.0} <music21.note.Note A>
Alto
{0.0} <music21.note.Note F>
Tenor
{0.0} <music21.note.Note D>
Bass
{0.0} <music21.note.Note D>
current bar: 1
Soprano
{0.0} <music21.note.Note A>
{1.0} <music21.note.Note F>
{2.0} <music21.note.Note G>
{3.0} <music21.note.Note A>
...
current bar: 4
Soprano
{0.0} <music21.note.Note F>
{1.0} <music21.note.Note G>
{2.0} <music21.note.Note A>
current bar: 4a        <-- same bar number, different suffix
Soprano
{0.0} <music21.note.Note A>
...
```

The full printout (and per-voice SATB extraction utilities) can be reproduced with the exploration cells in [`Notebooks/data_extraction.ipynb`](Notebooks/data_extraction.ipynb).

</details>

After filtering: **368 files found → 362 chorales kept → 29,795 chords**. The split is **grouped by chorale** — 262 train / 50 validation / 50 test with a fixed seed — so models are always evaluated on chorales they have never seen.
