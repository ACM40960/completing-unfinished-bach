# Running the Notebooks

The project runs on Google Colab or any Jupyter environment.

1. **Dependencies** — `music21`, `tensorflow`, `scikit-learn`, `pandas`, `seaborn` (preinstalled on Colab), plus for the audio notebook:
<br>

   ```
   apt-get install fluidsynth fluid-soundfont-gm
   pip install midi2audio
   ```
   <br>
   
2. **Data** — nothing to download manually: music21's Bach corpus ships with the library.
   
3. **Order** — `data_extraction` → `markov_analysis` → `model_sweeps` → `chained_model_evaluation` → `audio_rendering`. The sweeps are the slow part (54 model fits); the chained model alone trains in a few minutes.
4. **Reproducibility** — the chorale split uses a fixed seed, so train/validation/test membership is stable across runs. TensorFlow weight initialisation is not seeded, so metrics may differ slightly from the reported ones.
