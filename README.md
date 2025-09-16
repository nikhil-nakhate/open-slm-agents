Open SLM Agents — Modular GPT Repo

Overview

- Modular, config‑driven GPT implementations with a model registry (Detectron2/MMDet style).
- Clean separation of concerns: tokenizer, embeddings, transformer, output head, loss, datasets, logging.
- Hierarchical YAML configs with overrides via extends, per‑module freeze flags, and trainer settings.
- Simple trainer with checkpointing/resume, mixed precision, and optional W&B/TensorBoard logging.
- Interactive evaluation with greedy/sampling decode, and support for loading converted GPT‑2 weights.

Quick Start

- Install (Python 3.8+):
  - Base: `pip install -e .`
  - With optional deps (HF, W&B, TensorBoard): `pip install -e .[all]`

- Train:
  - `python train.py --mode pretraining --config base --logger none`
  - `python train.py --mode pretraining --config gpt2_base --logger tensorboard`
  - Resume: `python train.py --mode pretraining --config gpt2_base --resume outputs/gpt2-base/step_200.pt`

- Evaluate (interactive REPL):
  - `python eval.py --config gpt2_base`
  - With training checkpoint: `--checkpoint outputs/gpt2-base/step_200.pt`
  - With converted GPT‑2 weights: `--weights_dir weights/gpt2/355M`
  - Generation settings come from `eval` section in the config.

Project Structure

- `models/`
  - `__init__.py` — registry and registration helpers
  - `meta_arch/gpt.py` — GPT model class (from_config factory)
  - `modules/` — building blocks and builders
    - `build.py` — `build_*` helpers for tokenizer, embeddings, transformer, head, loss, norms
    - `embeddings.py` — `TokenEmbedding`, `PositionEmbedding`, and output projection
    - `transformer.py` — `TransformerBlock`, `Transformer`
    - `mha.py` — `MultiHeadAttention`
    - `layer_norm.py` — LayerNorm module
    - `losses.py` — loss builders (e.g., cross‑entropy)

- `ops/`
  - `config.py` — YAML loader with hierarchical extends
  - `tokenizer.py` — tokenizer base + simple/regex/HF GPT‑2 options

- `data/`
  - `dataset.py` — `BaseDataset`, `TextFileDataset`, and dataset+collate builder via config

- `metrics/`
  - `loggers/` — `BaseLogger`, `WandBLogger`, `TensorBoardLogger`

- `configs/`
  - `base.yaml`, `gpt2_base.yaml`, `gpt2_medium.yaml` — hierarchical, override‑friendly

- `scripts/`
  - `gpt_download3.py` — downloads GPT‑2 TF weights into `weights/gpt2/<size>` and saves converted params
  - `load_gpt_weights.py` — loads converted GPT‑2 weights into our GPT model

- Top level:
  - `train.py` — trainer CLI
  - `eval.py` — interactive generation CLI

Config Basics

- YAML files support `extends` with deep merge.
- Model section:
  - `model.name`: model key in the registry (`gpt`)
  - `model.modules.tokenizer`: tokenizer kind + params
  - `model.modules.token_embedding`, `position_embedding`, `emb_dropout`
  - `model.modules.transformer`:
    - `dim`, `n_layers`, `n_heads`, `mlp_mult`, `activation`, `qkv_bias`, `prenorm`
    - `context_length` defaults to `model.params.max_seq_len` if omitted
  - `model.modules.output_projection`: `tie_weights`
  - `model.modules.loss`: e.g., `kind: cross_entropy`
  - `model.params`: global params like `max_seq_len`, `dropout` (and inferred `vocab_size`)

- Train section:
  - Optimizer: `lr`, `betas`, `weight_decay`
  - Loop: `batch_size`, `max_steps`, `log_every`, `save_every`, `amp`
  - Checkpointing: `output_dir`, `resume`
  - Scheduler: `scheduler.kind` (linear|cosine|none), `warmup_steps`, `min_lr`
  - Data loader: `train.data_loader.kind` (e.g., `language_modeling_text`), `block_size`, `shuffle`, workers, etc.

- Eval section:
  - `max_new_tokens`, `temperature`, `top_k`, `top_p`, `greedy`

Add a New Model

- Create a class under `models/meta_arch/` and decorate with `@register_model("my_key")`.
- Implement `@classmethod from_config(cls, cfg)` that extracts config, prepares kwargs, and returns `cls(**kwargs)`.
- Compose submodules using `models/modules/build.py` helpers. Respect `freeze` flags via builder.

Datasets

- Default `TextFileDataset` reads `.txt` from `train.data_dir` and chunks into token windows.
- Crawler for remote JSON datasets (SFT/RL/RAG):
  - `python crawlers/sample_instruction_data.py --url <http_url> --category sft --filename data.json`
  - Saved into `data/<category>/filename`.

Converted GPT‑2 Weights

- Download/convert: `python scripts/gpt_download3.py` (saves into `weights/gpt2/<size>`)
- Eval with converted weights: `python eval.py --config gpt2_base --weights_dir weights/gpt2/355M`

Tips & Troubleshooting

- Missing YAML support: `pip install pyyaml`
- HF tokenizer: `pip install transformers` and use `model.modules.tokenizer.kind: hf_gpt2`
- TensorBoard/W&B: install extras `pip install -e .[tb]` or `[wandb]`
- TensorFlow is only required for the GPT‑2 download/conversion script.

License

- See repository license (if provided). If not specified, please clarify your preferred licensing.

