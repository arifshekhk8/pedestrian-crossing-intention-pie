"""prep_probs.py — collect the test-set probability vectors the figures need.

Writes `_probs.npz` next to this file: the shared ground-truth vector plus one
5-seed ensemble probability vector per model. Four come straight from the caches
written by the training runs; the bounding-box-only ablation has no cache, so it
is re-scored from its five stored checkpoints exactly as
journal_prep/Analysis/00_generate_analysis.py does.

Run once (about 20 s):  python prep_probs.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SEEDS = [42, 0, 1, 2, 3]

F1OPT = ROOT / "f1_optimization" / "probs_cache"
GRUC = ROOT / "gru" / "phase4_final" / "probs_cache"
RNNC = ROOT / "rnn" / "phase4_final" / "probs_cache"
MULTISEED = (ROOT / "journal_prep" / "issue2_clean_protocol" / "kaggle_result"
             / "runs_multiseed_clean")

CACHED = {
    "bilstm_f1": F1OPT / "lstm_a3_ens_test.npy",
    "bilstm_base": F1OPT / "lstm_frozen_ens_test.npy",
    "transformer_f1": F1OPT / "tf_b3_ens_test.npy",
    "transformer_searched": F1OPT / "tf_frozen_ens_test.npy",
    "gru_f1": GRUC / "gru_f1_winner_ens_test.npy",
    "rnn_f1": RNNC / "rnn_f1_winner_ens_test.npy",
}


class BBoxBiLSTM(nn.Module):
    """The 4-D (no ego-speed) clean-protocol variant's exact architecture."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(4, 64)
        self.bilstm = nn.LSTM(64, 128, 2, dropout=0.3, bidirectional=True,
                              batch_first=True)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        out, _ = self.bilstm(torch.relu(self.input_proj(x)))
        return self.head(out[:, -1, :])


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def regen_bbox_only(X_test):
    X = X_test[:, :, :4].astype(np.float32)
    acc = []
    for s in SEEDS:
        rd = MULTISEED / f"bilstm_bbox_only_seed{s}"
        mean, std = np.load(rd / "norm_mean.npy"), np.load(rd / "norm_std.npy")
        ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
        model = BBoxBiLSTM()
        model.load_state_dict(ck["model"])
        model.eval()
        Xn = ((X - mean) / std).astype(np.float32)
        acc.append(torch.sigmoid(model(torch.from_numpy(Xn)).squeeze(-1)).numpy())
    return np.mean(acc, axis=0)


def main():
    eng = _load("eng", ROOT / "journal_prep" / "issue12_unified_pipeline"
                / "12_unified_engine.py")
    *_, X_test, y_test = eng.load_splits()
    y_test = y_test.astype(int)

    out = {"y": y_test}
    for key, path in CACHED.items():
        p = np.load(path)
        assert p.shape == y_test.shape, f"{key}: {p.shape} vs {y_test.shape}"
        out[key] = p
    out["bilstm_bbox_only"] = regen_bbox_only(X_test)

    np.savez(HERE / "_probs.npz", **out)
    print(f"wrote {HERE / '_probs.npz'}")
    print(f"  n = {len(y_test)} windows, {y_test.mean():.4f} positive")
    for k, v in out.items():
        if k != "y":
            print(f"  {k:24s} mean p = {v.mean():.4f}")


if __name__ == "__main__":
    main()
