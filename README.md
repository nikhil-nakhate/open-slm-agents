<div align="center">

# Open SLM Agents — Build small agents from scratch 🚀

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](#) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](#)

<i>Config‑driven, registry‑based SLMs (incl. GPT) with clean modularity.</i>

<sub>
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#datasets-and-crawlers">Datasets</a> •
  <a href="#converted-gpt-2-weights">GPT-2 Weights</a>
</sub>

</div>

## ✨ Features

- Modular, config‑driven SLMs
- Separate builders for tokenizer, embeddings, transformer, projection, loss
- Hierarchical YAML configs (`extends`) with per‑module freeze flags
- Trainer with AMP, checkpoints/resume, schedulers, and logging (W&B/TensorBoard)
- Interactive eval REPL with greedy/sampling decode and GPT‑2 weight loading

---

<a id="installation"></a>
## 🔧 Installation

```bash
# Base install
pip install -e .

# With optional dependencies (HF + W&B + TensorBoard)
pip install -e .[all]
```

Requirements: Python 3.8+, PyTorch 2.0+. Optional: transformers, wandb, tensorboard, tensorflow (for GPT‑2 conversion).

---

<a id="quick-start"></a>
## 🚀 Quick Start

Train:
```bash
python train.py --mode pretraining --config base --logger none
python train.py --mode pretraining --config gpt2_base --logger tensorboard

# Resume from checkpoint
python train.py --mode pretraining --config gpt2_base --resume outputs/gpt2-base/step_200.pt
```

Evaluate (interactive REPL):
```bash
python eval.py --config gpt2_base
# Optional
python eval.py --config gpt2_base --checkpoint outputs/gpt2-base/step_200.pt
python eval.py --config gpt2_base --weights_dir weights/gpt2/355M
```

Generation settings are read from the `eval` section of your config.

---

<a id="configuration"></a>
## 🧩 Configuration

YAML configs support `extends` with deep merge. High‑level layout:

```yaml
model:
  name: gpt
  params:
    max_seq_len: 1024        # context length
    dropout: 0.1             # global dropout default
  modules:
    tokenizer: { kind: tiktoken }
    token_embedding: { freeze: false }
    position_embedding: { freeze: false }
    emb_dropout: { p: 0.1 }
    transformer:
      dim: 768
      n_layers: 12
      n_heads: 12
      mlp_mult: 4
      activation: gelu
      qkv_bias: false
      prenorm: true
    output_projection: { tie_weights: true }
    loss: { kind: cross_entropy, params: { ignore_index: -100 } }

train:
  lr: 2.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  batch_size: 4
  max_steps: 1000
  save_every: 200
  log_every: 20
  amp: true
  output_dir: outputs/gpt2-base
  scheduler: { kind: cosine, warmup_steps: 100, min_lr: 0.0 }
  data_loader: { kind: language_modeling_text, block_size: 1024, shuffle: true }

eval:
  max_new_tokens: 80
  temperature: 0.9
  top_k: 40
  top_p: 0.0
  greedy: false
```

Notes:
- `model.modules.transformer.dim/n_layers/n_heads` are the single source of truth.
- `vocab_size` is inferred from the tokenizer if omitted.
- Per‑module `freeze: true` is respected at build time (no trainer logic needed).

---

<a id="project-structure"></a>
## 🗂 Project Structure

```
models/
  __init__.py                 # registry
  meta_arch/gpt.py            # GPT (from_config)
  modules/
    build.py                  # builders (tokenizer/emb/blocks/head/loss/norm)
    embeddings.py             # TokenEmbedding, PositionEmbedding, OutputProjection
    transformer.py            # TransformerBlock/Transformer
    mha.py                    # MultiHeadAttention
    layer_norm.py             # LayerNorm
    losses.py                 # loss builders
ops/
  config.py                   # YAML loader + extends
  tokenizer.py                # tokenizers (simple/tiktoken/harmony)
data/
  dataset.py                  # BaseDataset, TextFileDataset, builder
metrics/
  loggers/                    # BaseLogger, WandBLogger, TensorBoardLogger
configs/
  base.yaml, gpt2_base.yaml, gpt2_medium.yaml
scripts/
  gpt_download3.py            # download + save GPT‑2 params
  load_gpt_weights.py         # load converted params into our GPT
train.py                      # trainer CLI
eval.py                       # interactive eval CLI
```

---

<a id="datasets-and-crawlers"></a>
## 📦 Datasets & Crawlers

### HuggingFace Datasets

Download datasets from HuggingFace Hub:

```bash
# Download dataset
python crawlers/download_hf_dataset.py \
  --repo-id "tatsu-lab/alpaca" \
  --category sft

# Download specific split
python crawlers/download_hf_dataset.py \
  --repo-id "tatsu-lab/alpaca" \
  --category sft \
  --split train
```

### HuggingFace Models

Download model weights from HuggingFace Hub:

```bash
# Download entire model
python crawlers/download_hf_model.py --repo-id "gpt2"

# Download specific file
python crawlers/download_hf_model.py \
  --repo-id "gpt2" \
  --filename "pytorch_model.bin"
```

### Remote URL Downloads

Download from direct URLs (SFT/RL/RAG):

```bash
python crawlers/sample_instruction_data.py \
  --url https://.../instruction-data.json \
  --category sft \
  --filename instruction-data.json
```

**Note:** For private HuggingFace datasets/models, set `HUGGINGFACE_TOKEN` in your `.env` file.

Datasets saved to `data/<category>/<filename>`, models to `weights/<model-name>/`.

---

<a id="converted-gpt-2-weights"></a>
## 📥 Model Weights

### GPT-2 Weights (Converted)

```bash
# Download + convert
python scripts/gpt_download3.py

# Evaluate with converted weights
python eval.py --config gpt2_base --weights_dir weights/gpt2/355M
```

The loader maps GPT‑2 tensors into our module layout (token/pos embeddings, QKV, MLP, norms, head).

### GPT-OSS 20B Official Weights

Load the official GPT-OSS weights from OpenAI's HuggingFace repository and run them through the refactored modular implementation:

```bash
# Set your HuggingFace token in .env first
echo "HUGGINGFACE_TOKEN=your_token_here" >> .env

# Download the safetensors checkpoint
python scripts/load_gpt_oss_weights.py --repo-id openai/gpt-oss-20b --output-dir weights/gpt-oss-20b

# Update configs/models/gpt_oss_20b.yaml and point `model.weights`
# to the directory that now contains model.safetensors, then run:
python infer.py --config configs/models/gpt_oss_20b.yaml
```

**Features:**
- 21B parameters (3.6B active with MoE)
- Mixture of Experts with 32 experts, top-4 routing
- Grouped Query Attention (64 heads, 8 KV heads)
- RoPE with YaRN scaling for extended context (131K tokens)
- Sliding window attention (128 tokens on alternating layers)
- MXFP4 quantized weights for efficient inference

---

## 🛠 Tips & Troubleshooting

- YAML: `pip install pyyaml`
- Tokenizers: `pip install tiktoken` for tiktoken/harmony tokenizers
- HuggingFace: `pip install huggingface-hub datasets` for downloading HF datasets/models
- Logging: `pip install -e .[tb]` or `pip install -e .[wandb]`
- TensorFlow is only needed for GPT‑2 download/conversion scripts.
- Environment: Copy `.env.example` to `.env` and add your API tokens (HuggingFace, OpenAI, etc.)

---

## 📄 License

Please include your license of choice for this repository.
