import os
from typing import List, Dict, Any, Callable, Tuple

import torch
from torch.utils.data import Dataset


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

    if kind in {"language_modeling_text", "lm_text"}:
        ds_cfg = {
            "data_dir": cfg.get("train", {}).get("data_dir", "data"),
            "min_length": dl_cfg.get("min_length", 32),
        }
        dataset = TextFileDataset(ds_cfg, tokenizer)
        block_size = dl_cfg.get("block_size", cfg.get("model", {}).get("params", {}).get("max_seq_len", 128))
        pad_id = getattr(tokenizer, "pad_id", 0)
        collate = lambda batch: _chunking_collate(batch, block_size=block_size, pad_id=pad_id)
        return dataset, collate

    raise ValueError(f"Unknown data_loader kind: {kind}")

