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
