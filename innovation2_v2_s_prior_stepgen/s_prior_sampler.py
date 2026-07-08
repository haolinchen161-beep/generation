"""无条件 S 先验采样器。

当前先跑通版本采用经验分布采样：从 DeepCAD-30 S-ready 训练集的 S prior 中
随机抽样得到 S*。这仍然是无条件生成，因为用户不提供目标 S；后续可以把本模块
替换成神经 S prior 模型。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils_io import read_jsonl, write_jsonl


class EmpiricalSPriorSampler:
    """从训练集 S prior 经验分布中采样 S*。"""

    def __init__(
        self,
        prior_jsonl: Path,
        split_uids_path: Path = None,
        seed: int = 20260708,
        min_faces: int = 0,
        max_faces: Optional[int] = None,
        max_edges: Optional[int] = None,
    ) -> None:
        self.prior_jsonl = Path(prior_jsonl)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.rows = read_jsonl(self.prior_jsonl)
        if split_uids_path is not None and Path(split_uids_path).exists():
            allowed = set(Path(split_uids_path).read_text(encoding="utf-8").splitlines())
            self.rows = [row for row in self.rows if row.get("uid") in allowed]
        self.rows = [
            row
            for row in self.rows
            if int(row.get("num_faces", 0) or 0) >= int(min_faces)
            and (max_faces is None or int(row.get("num_faces", 0) or 0) <= int(max_faces))
            and (max_edges is None or int(row.get("num_edges", 0) or 0) <= int(max_edges))
        ]
        if not self.rows:
            raise RuntimeError(f"S prior 采样池为空: {self.prior_jsonl}")

    def sample(self, num_samples: int) -> List[Dict[str, Any]]:
        sampled: List[Dict[str, Any]] = []
        for idx in range(int(num_samples)):
            row = dict(self.rng.choice(self.rows))
            row["sample_id"] = f"s_uncond_{idx:05d}"
            row["sample_source_uid"] = row.get("uid", "")
            row["uid"] = row["sample_id"]
            row["unconditional_prior_sample"] = True
            row["sampling_policy"] = "empirical_bootstrap_from_training_S_prior"
            sampled.append(row)
        return sampled


def sample_and_save(
    prior_jsonl: Path,
    split_uids_path: Path,
    output_path: Path,
    num_samples: int,
    seed: int,
    min_faces: int = 0,
    max_faces: Optional[int] = None,
    max_edges: Optional[int] = None,
) -> List[Dict[str, Any]]:
    sampler = EmpiricalSPriorSampler(
        prior_jsonl=prior_jsonl,
        split_uids_path=split_uids_path,
        seed=seed,
        min_faces=min_faces,
        max_faces=max_faces,
        max_edges=max_edges,
    )
    rows = sampler.sample(num_samples)
    write_jsonl(output_path, rows)
    return rows
