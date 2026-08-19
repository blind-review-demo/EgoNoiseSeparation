# Learning2HearWhileWalking

This repository contains the public EgoGraph and Transfer-DiT implementations
used for ego-noise separation while walking. It does not vendor PE-AV or
SAM-Audio source code or model weights.

## What is included

- `learning2hear.run_egograph`: self-anchored graph propagation over PE-AV
  audio embeddings, run independently for each robot.
- `learning2hear.models.TransferDiT`: a prompt-free SAM-Audio separator.
- Bottleneck adapter layers at the original DiT cross-attention residual site,
  plus the rank-16 LoRA modules used by the released checkpoints.
- Setup scripts for cloning SAM-Audio and downloading the SAM-Audio base model.
- A minimal inference script for local audio files.

## What is not included

- SAM-Audio source code.
- SAM-Audio model weights.
- Private data, evaluation splits, and paper-generation scripts.

Fine-tuned G1 and Go1 checkpoints are not included in this anonymous release.

## License

This repository is released under the Creative Commons Attribution 4.0
International License.

SAM-Audio code and weights are third-party assets. Check and follow their own
license terms before use:

- SAM-Audio code: https://github.com/facebookresearch/sam-audio
- SAM-Audio small weights: https://huggingface.co/facebook/sam-audio-small

## Installation

Create and activate a conda environment, then install this package:

```bash
conda create -n l2hww python=3.11 -y
conda activate l2hww
python -m pip install -e .
```

Clone and install SAM-Audio:

```bash
bash scripts/setup_sam_audio.sh
```

Log in to Hugging Face and download the base SAM-Audio checkpoint. The
`facebook/sam-audio-small` model may require accepting the model access terms on
Hugging Face first.

```bash
huggingface-cli login
python scripts/download_sam_audio_weights.py --model-id facebook/sam-audio-small
```

## Inference

Run Transfer-DiT with the SAM-Audio base checkpoint:

```bash
python scripts/separate_audio.py \
  --audio path/to/mixed.wav \
  --out-dir outputs/demo \
  --description "walking ego-noise"
```

To use a fine-tuned Transfer-DiT checkpoint, pass it explicitly:

```bash
python scripts/separate_audio.py \
  --audio path/to/mixed.wav \
  --checkpoint checkpoints/transfer_dit.pt \
  --out-dir outputs/demo
```

Transfer-DiT ignores text, video, and span prompts internally. The
`--description` argument is kept only to satisfy SAM-Audio processor batching.

## EgoGraph

EgoGraph first embeds every unlabeled adaptation clip with PE-AV. For each
robot, it forms a self-anchor from the normalized mean embedding, uses the upper
and lower fractions `rho = 0.10` as positive and negative seeds, and propagates
the seed scores over a 64-nearest-neighbor graph in a 128-dimensional PCA
space. Clips with a propagated score greater than `0.90` are selected for
Transfer-DiT training.

Set up the official PE-AV implementation and extract embeddings:

```bash
bash scripts/setup_pe_av.sh
python scripts/extract_pe_av_embeddings.py \
  --manifest path/to/adaptation_manifest.csv \
  --output outputs/pe_av_embeddings.npz
```

The manifest must contain `mix_path` and `robot` columns. Run EgoGraph:

```bash
python scripts/run_egograph.py \
  --manifest path/to/adaptation_manifest.csv \
  --embeddings outputs/pe_av_embeddings.npz \
  --output outputs/egograph_scores.csv
```

The output contains the self-anchor similarity, initial seed label, propagated
ego-noise-dominant score, and training-selection decision for every clip. PE-AV
is used only to obtain embeddings; EgoGraph itself uses no text prompt.

## Repository layout

```text
learning2hear/
  config.py
  egograph.py
  models/transfer_dit.py
scripts/
  setup_pe_av.sh
  extract_pe_av_embeddings.py
  run_egograph.py
  setup_sam_audio.sh
  download_sam_audio_weights.py
  separate_audio.py
```
