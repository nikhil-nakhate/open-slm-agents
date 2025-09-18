import os
import json
from glob import glob
from functools import partial
from typing import List, Dict, Any, Callable, Tuple

import torch
from torch.utils.data import Dataset
from data.transforms import build_text_pair_transform


class BaseDataset(Dataset):
    """Base class for datasets to enable a uniform interface."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def __len__(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def __getitem__(self, idx: int):  # pragma: no cover - interface
        raise NotImplementedError


class TextFileDataset(BaseDataset):
    """Reads .txt files under a directory and returns token id lists (one per file)."""

    def __init__(self, cfg: Dict[str, Any], tokenizer):
        super().__init__(cfg)
        data_dir = cfg.get("data_dir", "data")
        min_length = int(cfg.get("min_length", 32))
        texts: List[str] = []
        if os.path.isdir(data_dir):
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.endswith(".txt"):
                        with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as fh:
                            texts.append(fh.read())
        if not texts:
            texts = [
                "Hello world. This is a tiny corpus for quick smoke testing.",
                "Building a modular GPT repository with configs and registry.",
            ]
        ids_lists = [tokenizer.encode(t) for t in texts]
        self.ids: List[List[int]] = [ids for ids in ids_lists if len(ids) >= min_length] or ids_lists

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> List[int]:
        return self.ids[idx]


def _chunking_collate(batch: List[List[int]], block_size: int, pad_id: int = 0) -> Dict[str, torch.Tensor]:
    # Batch is a list of variable-length token id lists. We generate fixed-size windows.
    xs, ys = [], []
    for ids in batch:
        if len(ids) < block_size + 1:
            pad = [pad_id] * (block_size + 1 - len(ids))
            seq = pad + ids
        else:
            seq = ids[: block_size + 1]
        x = seq[:-1]
        y = seq[1:]
        xs.append(x)
        ys.append(y)
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return {"input_ids": x, "labels": y}


class SFTJsonDataset(BaseDataset):
    """Supervised fine-tuning dataset over JSON files.

    - Expects each JSON file to contain a list of records (dict).
    - A text-pair transform converts each record to (prompt, target) strings.
    - Encodes (prompt + target) into a single list of token ids using the provided tokenizer.
    """

    def __init__(self, cfg: Dict[str, Any], paths: List[str], transform: Callable[[Dict], Tuple[str, str]], tokenizer):
        super().__init__(cfg)
        self.paths = paths
        self.transform = transform
        self.tokenizer = tokenizer
        sequences: List[List[int]] = []
        for p in self.paths:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        prompt, target = self.transform(rec)
                        ids = self.tokenizer.encode(prompt) + self.tokenizer.encode(target)
                        sequences.append(ids)
        self.items = sequences

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> List[int]:
        return self.items[idx]


def _sft_collate_batch(batch: List[List[int]], pad_token_id: int, ignore_index: int, allowed_max_length: int) -> Dict[str, torch.Tensor]:
    """Custom collate function from notebook - handles instruction fine-tuning data properly."""
    # Find the longest sequence length in this batch (+1 for added pad/eos token)
    batch_max_length = max((len(seq) + 1) for seq in batch) if batch else 0

    inputs_lst: List[torch.Tensor] = []
    targets_lst: List[torch.Tensor] = []

    for seq in batch:
        new_item = list(seq)
        # Add an <|endoftext|>/pad token at the end
        new_item += [pad_token_id]
        # Pad to batch max length
        pad_len = batch_max_length - len(new_item)
        if pad_len > 0:
            new_item = new_item + [pad_token_id] * pad_len

        inputs = torch.tensor(new_item[:-1], dtype=torch.long)
        targets = torch.tensor(new_item[1:], dtype=torch.long)

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if torch.numel(indices) > 1:
            targets[indices[1:]] = ignore_index

        # Truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst) if inputs_lst else torch.empty(0, dtype=torch.long)
    targets_tensor = torch.stack(targets_lst) if targets_lst else torch.empty(0, dtype=torch.long)

    return {"input_ids": inputs_tensor, "labels": targets_tensor}


def build_dataset_and_collate(cfg: Dict[str, Any], tokenizer) -> Tuple[Dataset, Callable]:
    """Build dataset and collate_fn from train.data_loader config.

    Expects:
    train:
      data_loader:
        kind: language_modeling_text
        block_size: 128
        num_workers: 0
        shuffle: true
    """
    dl_cfg = cfg.get("train", {}).get("data_loader", {})
    kind = dl_cfg.get("kind", "language_modeling_text").lower()
    print(f"In build_dataset_and_collate with kind: {kind}")
    if kind in {"language_modeling_text", "lm_text"}:
        ds_cfg = {
            "data_dir": cfg.get("train", {}).get("data_dir", "data"),
            "min_length": dl_cfg.get("min_length", 32),
        }
        dataset = TextFileDataset(ds_cfg, tokenizer)
        block_size = dl_cfg.get("block_size", cfg.get("model", {}).get("params", {}).get("max_seq_len", 128))
        pad_id = getattr(tokenizer, "pad_id", 0)
        collate = partial(_chunking_collate, block_size=block_size, pad_id=pad_id)
        return dataset, collate

    if kind in {"sft_json", "sft"}:
        data_dir = cfg.get("train", {}).get("data_dir", os.path.join("data", "sft"))
        pattern = dl_cfg.get("pattern", "*.json")
        paths = sorted(glob(os.path.join(data_dir, pattern)))
        if not paths:
            data_dir = os.path.join("data", "sft")
            paths = sorted(glob(os.path.join(data_dir, pattern)))
        # Prefer top-level data.transforms.template, fallback to data_loader.transform
        data_transforms = cfg.get("data", {}).get("transforms", {})
        template = data_transforms.get("template")
        if template:
            transform_cfg = {"kind": template}
        else:
            transform_cfg = dl_cfg.get("transform", {"kind": "alpaca"})
        transform = build_text_pair_transform(transform_cfg)

        dataset = SFTJsonDataset({"data_dir": data_dir}, paths, transform, tokenizer)
        print(f"Created SFT dataset with {len(dataset)} samples")

        block_size = dl_cfg.get("block_size", cfg.get("model", {}).get("params", {}).get("max_seq_len", 1024))
        ignore_index = cfg.get("model", {}).get("modules", {}).get("loss", {}).get("params", {}).get("ignore_index", -100)
        pad_token_id = getattr(tokenizer, "pad_id", 50256)  # GPT-2 pad token ID
        collate = partial(_sft_collate_batch, pad_token_id=pad_token_id, ignore_index=ignore_index, allowed_max_length=block_size)
        return dataset, collate

    raise ValueError(f"Unknown data_loader kind: {kind}")
