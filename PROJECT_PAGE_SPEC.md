# Project Page Specification

## Purpose

This static page presents audio examples and real-world videos for the
RecurGraph and Transfer-DiT experiment. The experiment and page-builder source
of truth is the private experiment repository; generated publication assets
are written to this GitHub Pages repository. The repositories have separate
source and publication roles and must not be treated as interchangeable copies.

Overall evaluation tables report method-level means by robot and SNR, omit
`clean_only`, round displayed values to two decimals, and bold the best value in
each robot-SNR metric column. Each audio example set compares the mixture,
references, and separated target estimates with spectrograms.

## Source Data

The project page builder uses:

- Sample-level results configured by
  `Paths.cross_site_v3_all_method_sample_level_results_csv`.
- Overall evaluation scores configured by
  `Paths.cross_site_v3_all_method_scores_long_csv`.

Both paths are defined in the private experiment repository's `src/config.py`.

## Generated Files

The builder writes or refreshes:

- `assets/data/project-data.json`
- `assets/spectrogram_colorbar.png`
- `assets/audio/<sample_id>/*.wav`
- `assets/audio/<sample_id>/*.png`
- `assets/real_exp_videos/<sample_id>/poster.jpg`

The repository root contains the static page shell (`index.html`, `style.css`, and
`script.js`) and is the GitHub Pages publish directory.

## Audio Example Set Layout

Each set shows metadata at the top:

- `Robot: Go1` or `Robot: G1`
- `SNR: <value> dB`
- `Class: <class name>`
- `Ground Truth: Normal` or `Ground Truth: Anomaly`

The upper audio row contains:

- `Mixture`
- `Ego-Noise`
- `Target`

The lower audio row contains method outputs:

- `EgoSep (Ours)`
- `SAM-Audio` (`Sam-Audio`)
- `CLAPSep` (`clapsep`)
- `Conv-TasNet` (`espnet_convtasnet`)
- `Dual-Path RNN` (`espnet_dprnn_tf`)
- `Conformer` (`espnet_conformer`)
- `SB-INMF` (`sb_inmf_gp`)
- `Harmonic NMF` (`oracle_harmonic_nmf`)

Each method output also displays:

- `Prediction: Normal` or `Prediction: Anomaly`
- `Correct` or `Wrong`
- `CLAP Score↑`: CLAP audio embedding cosine similarity between the target and
  the separated estimate.
- `SAJ Score↑`: overall SAJ separation judgment score.

## Sample Selection

The builder selects 12 sets, as defined by `ProjectPage.sample_count`.
Candidates are restricted to 0 dB by `ProjectPage.sample_snr_dbs`.

Selection is automatic and prioritizes samples where:

- `EgoSep` has a correct binary Normal/Anomaly prediction.
- At least one displayed competing method has a wrong binary prediction.
- Environmental sound categories remain varied.
- More displayed competing methods have wrong binary predictions.
- `EgoSep` has higher separation metrics than competing methods.

The display alternates according to `ProjectPage.sample_bucket_cycle`:

1. `G1 / Anomaly`
2. `Go1 / Normal`

The cycle repeats until 12 sets have been selected, so both Robot and Ground
Truth alternate on every displayed set. The chosen orientation has more
competitor failures in the current source results than the opposite alternating
orientation. `Correct` / `Wrong` is derived from the displayed binary prediction
and Ground Truth, never from the 52-class correctness field.

## Audio Scale

Audio files keep the source scale when they are materialized for the project page.

The builder loads each source track and writes it to `assets/audio/`
without peak normalization or any other amplitude scaling.

## Spectrograms

Every playable audio file has one spectrogram image directly above it.

Spectrogram settings are defined only in `src/config.py`:

- `ProjectPage.spectrogram_n_fft`
- `ProjectPage.spectrogram_hop_length`
- `ProjectPage.spectrogram_percentile_cut`
- `ProjectPage.spectrogram_cmap`

Current settings:

- color limits: per-example mixture spectrogram `5%` to `95%` percentile
- colormap: `jet`
- x-axis unit: `sec`
- y-axis unit: `kHz`
- axis label and tick text is kept compact but readable for embedded images

The same percentile-derived limits are used for mixture, ego-noise, target, and all method outputs in each example.

A single shared colorbar is placed near the top of the page.

## Build Command

Run:

```bash
python scripts/build_project_page.py
```

This regenerates:

- selected samples
- wav files at the source scale
- spectrogram images
- colorbar
- JSON payload

## Local Preview

Run:

```bash
python -m http.server 8000 --directory /path/to/project-page
```

Open:

```text
http://127.0.0.1:8000/
```

## Configuration Rule

Page-generation parameters must be defined in `src/config.py`.
Scripts must import parameters from `src/config.py` rather than defining them
inline.

## Privacy Rule

Published files must not contain personal names, personal account identifiers,
email addresses, local absolute paths, or geolocation and capture-time metadata.
Video assets must be remuxed with global, stream, and chapter metadata removed
before publication.
