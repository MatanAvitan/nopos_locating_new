"""Shared code for the complete-circuit study (mechanism_report.md, brief of Oct 2026).

One `Net` wraps any pre-LN NoPE checkpoint used here (2-layer position models
and the 6-layer LM) and runs a manual forward from the weights. Every stage
passes through `Hook.__call__(stage, layer, tensor)`, which can capture or
replace it. Stage names, per block l:

    resid_in   [B,T,d]      block input residual h^(l-1)
    x_attn     [B,T,d]      attention-input LayerNorm output x^(l)
    q, k, v    [B,H,T,dh]   per-head projections (biases included)
    scores     [B,H,T,T]    scaled scores, -inf above the diagonal
    weights    [B,H,T,T]    softmax weights
    z          [B,H,T,dh]   per-head weighted values (sum_j a_ij v_j)
    attn_out   [B,T,d]      sum_h W_O,h z_h + b_O  (the attention update o^(l))
    resid_mid  [B,T,d]      h + o
    x_mlp      [B,T,d]      MLP-input LayerNorm output
    mlp_pre    [B,T,4d]     c_fc output (before the nonlinearity)
    mlp_out    [B,T,d]      MLP update
    resid_out  [B,T,d]      block output h^(l)

After the blocks: final_ln [B,T,d] and `out` (position prediction in absolute
positions for the position models, logits for the LM).

Blocks are 0-indexed in code. The brief's "layer 1" is block 0.
"""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MP = REPO_ROOT / "analysis_scripts" / "mechanism_pipeline"
sys.path.insert(0, str(MP))
from common import (BOS_TOKEN_ID, CHECKPOINTS, NANOGPT, VAL_BIN,  # noqa: E402
                    build_splits, head_weights, resolve_checkpoint)
import beta_common  # noqa: E402,F401  (registers the sweep and seed checkpoints)

OUT_ROOT = REPO_ROOT / "results" / "mechanism" / "complete_circuit"
EPS_LN = 1e-5

# ---------------------------------------------------------------------------
# Model registry. family: 1h / 12h / lm, plus controls. "target" marks the
# three architectures the brief asks about; the rest are labeled controls.
# ---------------------------------------------------------------------------

CHECKPOINTS["sw_L1024_unifL1_s2"] = NANOGPT / "out-sweep-1H-L1024-unifL1-s2/run/best_ckpt.pt"
CHECKPOINTS["sw_L1024_noMLP1_s2"] = NANOGPT / "out-sweep-1H-L1024-noMLP1-s2/run/best_ckpt.pt"
CHECKPOINTS["seed_12h_s1"] = NANOGPT / "out-mechanism-12H-1024-seed1/run/best_ckpt.pt"


@dataclass
class Spec:
    key: str
    kind: str            # "toy" or "lm"
    ckpt: str            # CHECKPOINTS key (toy) or path relative to nanoGPT (lm)
    L: int               # evaluation length (= training length)
    family: str          # "1h", "12h", "lm", "lmbos", or a control family
    role: str            # "target" or "control"
    bos: bool            # prepend token 50256 at position 0
    note: str = ""


SPECS = {s.key: s for s in [
    # two-layer, one head, fully trained (ATTN2-FULL), L = 1024
    Spec("1h_s42", "toy", "attn2_full", 1024, "1h", "target", True, "ATTN2-FULL lr3e-4, the paper's 1H model"),
    Spec("1h_s3", "toy", "seed_1h_s3", 1024, "1h", "target", True, "out-mechanism-1H-1024-seed3"),
    Spec("1h_sw_s1", "toy", "sw_L1024_s1", 1024, "1h", "target", True, "sweep L1024 seed 1"),
    Spec("1h_sw_s2", "toy", "sw_L1024_s2", 1024, "1h", "target", True, "sweep L1024 seed 2"),
    # two-layer, 12 heads, fully trained (FULL-12H), L = 1024
    Spec("12h_s42", "toy", "full12h", 1024, "12h", "target", True, "R0 nuacla0w"),
    Spec("12h_s1", "toy", "seed_12h_s1", 1024, "12h", "target", True, ""),
    Spec("12h_s2", "toy", "seed_12h_s2", 1024, "12h", "target", True, ""),
    Spec("12h_s3", "toy", "seed_12h_s3", 1024, "12h", "target", True, ""),
    # six-layer NoPE LM, native input (no inserted BOS)
    Spec("lm", "lm", "out-lm-6layer-fulltrain-ddp/ckpt.pt", 128, "lm", "target", False, "trained on raw OWT windows"),
    # controls and comparisons
    # LM analyses use the BOS-trained LM (user decision, Oct 2026). The earlier
    # "lm_bosin" condition (BOS prepended at test time to the native LM) is dropped.
    Spec("lmbos_s42", "lm", "out-lm-6layer-fulltrain-bos/ckpt.pt", 128, "lmbos", "target", True, "BOS-trained LM"),
    Spec("lmbos_s1", "lm", "out-lm-6layer-fulltrain-bos-seed1/ckpt.pt", 128, "lmbos", "target", True, ""),
    Spec("lmbos_s2", "lm", "out-lm-6layer-fulltrain-bos-seed2/ckpt.pt", 128, "lmbos", "target", True, ""),
    Spec("lmbos_s3", "lm", "out-lm-6layer-fulltrain-bos-seed3/ckpt.pt", 128, "lmbos", "target", True, ""),
    Spec("attn2_1h", "toy", "attn2_1h", 1024, "attn2_1h", "control", True, "ATTN2-1H (R2): untrained block 0"),
    Spec("unifL1_s42", "toy", "sw_L1024_unifL1_s42", 1024, "1h_unifL1", "control", True, "block-0 Q/K fixed at 0"),
    Spec("unifL1_s1", "toy", "sw_L1024_unifL1_s1", 1024, "1h_unifL1", "control", True, ""),
    Spec("unifL1_s2", "toy", "sw_L1024_unifL1_s2", 1024, "1h_unifL1", "control", True, ""),
    Spec("noMLP1_s42", "toy", "sw_L1024_noMLP1_s42", 1024, "1h_noMLP1", "control", True, "block 0 has no MLP"),
    Spec("noMLP1_s1", "toy", "sw_L1024_noMLP1_s1", 1024, "1h_noMLP1", "control", True, ""),
    Spec("noMLP1_s2", "toy", "sw_L1024_noMLP1_s2", 1024, "1h_noMLP1", "control", True, ""),
    Spec("relu_s42", "toy", "sw_L1024_relu_s42", 1024, "1h_relu", "control", True, "ReLU MLPs"),
    Spec("relu_s1", "toy", "sw_L1024_relu_s1", 1024, "1h_relu", "control", True, ""),
    Spec("attn1_s42", "toy", "h1_1layer_s42", 1024, "1layer", "control", True, "one-layer 1H"),
    Spec("attn1_s1", "toy", "h1_1layer_s1", 1024, "1layer", "control", True, ""),
]}

TARGETS = [k for k, s in SPECS.items() if s.role == "target"]


def ckpt_path(spec: Spec) -> Path:
    if spec.kind == "toy":
        return resolve_checkpoint(spec.ckpt)
    return NANOGPT / spec.ckpt


def sha256_full(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_state():
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
        return {"commit": head, "dirty_files": len(dirty.splitlines())}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_net(key, device):
    spec = SPECS[key]
    path = ckpt_path(spec)
    if spec.kind == "toy":
        from common import load_model
        model, meta = load_model(spec.ckpt, device)
        cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
        return Net(spec, model, device, dict(cfg), meta), path
    from lm_mechanism import load_lm
    model, cfg, ck = load_lm(str(path), device)
    meta = {"iter_num": ck.get("iter_num"), "best_val_loss": float(ck.get("best_val_loss", float("nan")))}
    train_cfg = dict(ck.get("config", {}))
    return Net(spec, model, device, {**dict(ck["model_args"]), "_train": train_cfg}, meta), path


@dataclass
class BlockRef:
    ln_1: torch.nn.Module
    attn: torch.nn.Module
    ln_2: object          # None when the block has no MLP-input LN
    mlp: torch.nn.Module
    skip_mlp: bool


class Net:
    """Uniform access to a pre-LN NoPE checkpoint."""

    def __init__(self, spec, model, device, cfg, meta):
        self.spec, self.model, self.device, self.cfg, self.meta = spec, model, device, cfg, meta
        m = model
        if spec.kind == "toy":
            raw = [m.block1] + ([m.block2] if m.config.n_layer == 2 else [])
            self.blocks = [BlockRef(b.ln_1, b.attn, b.ln_2, b.mlp, bool(getattr(b, "skip_mlp", False)))
                           for b in raw]
            self.wte, self.ln_f = m.wte, m.ln_f
            self.L_train = m.config.block_size
        else:
            self.blocks = [BlockRef(b.ln_1, b.attn, b.ln_2 if getattr(b, "use_ln2", True) else None,
                                    b.mlp, False) for b in m.transformer.h]
            self.wte, self.ln_f = m.transformer.wte, m.transformer.ln_f
            self.L_train = m.config.block_size
        self.n_layer = len(self.blocks)
        self.H = self.blocks[0].attn.n_head
        self.d = self.blocks[0].attn.n_embd
        self.dh = self.d // self.H
        self.hw = [head_weights(b.attn) for b in self.blocks]

    # -- per-block maps ----------------------------------------------------
    def ln_params(self, mod):
        """(gamma, beta) of a LayerNorm; beta is zeros when the LN has no bias."""
        g = mod.weight.detach().double()
        b = mod.bias.detach().double() if mod.bias is not None else torch.zeros_like(g)
        return g, b

    def ov(self, l, dtype=torch.float64):
        """B_OV per head [H,d,d] = W_O,h W_V,h."""
        hw = self.hw[l]
        return torch.einsum("hde,hef->hdf", hw.W_o.to(dtype), hw.W_v.to(dtype))

    def attn_bias(self, l, dtype=torch.float64):
        """b_attn = sum_h W_O,h b_V,h + b_O (added once)."""
        hw = self.hw[l]
        return self.head_bias(l, dtype).sum(0) + hw.b_o.to(dtype)

    def head_bias(self, l, dtype=torch.float64):
        """W_O,h b_V,h per head [H,d]."""
        hw = self.hw[l]
        return torch.einsum("he,hde->hd", hw.b_v.to(dtype), hw.W_o.to(dtype))

    # -- data --------------------------------------------------------------
    def batch(self, offsets):
        data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
        L = self.spec.L
        seqs = []
        for off in offsets:
            if self.spec.bos:
                seqs.append(np.concatenate([[BOS_TOKEN_ID], data[off:off + L - 1].astype(np.int64)]))
            else:
                seqs.append(data[off:off + L].astype(np.int64))
        return torch.from_numpy(np.stack(seqs)).to(self.device)

    # -- forward -----------------------------------------------------------
    @torch.no_grad()
    def native(self, idx):
        """The checkpoint's own forward. Position models: absolute positions.
        LM: logits."""
        if self.spec.kind == "toy":
            out, _ = self.model(idx, targets=None, capture_taps=False)
            return out.squeeze(-1).float() * (self.L_train - 1)
        m = self.model
        x = m.transformer.wte(idx)
        for b in m.transformer.h:
            x = b(x)
        return m.lm_head(m.transformer.ln_f(x)).float()

    @torch.no_grad()
    def forward(self, idx, hook=None, stop_after=None):
        """Manual forward. `hook(stage, l, t) -> t` may capture or edit.
        Returns the final output (see module docstring)."""
        E = hook if hook is not None else (lambda s, l, t: t)
        h = self.wte(idx).float()
        T = idx.shape[1]
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        for l, b in enumerate(self.blocks):
            hw = self.hw[l]
            h = E("resid_in", l, h)
            x = E("x_attn", l, F.layer_norm(h, (self.d,), b.ln_1.weight, b.ln_1.bias, EPS_LN))
            q = E("q", l, torch.einsum("btd,hed->bhte", x, hw.W_q.float()) + hw.b_q.float()[None, :, None])
            k = E("k", l, torch.einsum("btd,hed->bhte", x, hw.W_k.float()) + hw.b_k.float()[None, :, None])
            v = E("v", l, torch.einsum("btd,hed->bhte", x, hw.W_v.float()) + hw.b_v.float()[None, :, None])
            s = E("scores", l, (torch.einsum("bhte,bhse->bhts", q, k) * hw.scale).masked_fill(mask, float("-inf")))
            a = E("weights", l, torch.softmax(s, -1))
            z = E("z", l, torch.einsum("bhts,bhse->bhte", a, v))
            o = E("attn_out", l, torch.einsum("bhte,hde->btd", z, hw.W_o.float()) + hw.b_o.float())
            hm = E("resid_mid", l, h + o)
            if b.skip_mlp:
                m = torch.zeros_like(hm)
            else:
                xm = hm if b.ln_2 is None else F.layer_norm(hm, (self.d,), b.ln_2.weight, b.ln_2.bias, EPS_LN)
                xm = E("x_mlp", l, xm)
                pre = E("mlp_pre", l, b.mlp.c_fc(xm))
                m = E("mlp_out", l, b.mlp.c_proj(b.mlp.gelu(pre)))
            h = E("resid_out", l, hm + m)
            if stop_after is not None and l == stop_after:
                return h
        f = E("final_ln", self.n_layer, self.ln_f(h))
        if self.spec.kind == "toy":
            return E("out", self.n_layer, self.model.pos_head(f).squeeze(-1).float() * (self.L_train - 1))
        return E("out", self.n_layer, self.model.lm_head(f).float())

    def lm_loss(self, idx, logits):
        """Per-token next-token CE [B,T-1]."""
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                               idx[:, 1:].reshape(-1), reduction="none").reshape(idx.shape[0], -1)


class Capture:
    """Hook that records chosen (stage, layer) tensors and applies optional edits."""

    def __init__(self, want=None, edits=None):
        self.want = want          # set of (stage, layer) or None for all
        self.edits = edits or {}  # (stage, layer) -> fn(t) -> t
        self.store = {}

    def __call__(self, stage, l, t):
        fn = self.edits.get((stage, l))
        if fn is not None:
            t = fn(t)
        if self.want is None or (stage, l) in self.want:
            self.store[(stage, l)] = t
        return t


# ---------------------------------------------------------------------------
# Splits: calibration / validation reuse earlier (exploratory) windows; the
# test split is fresh and token-disjoint from every earlier window at L.
# ---------------------------------------------------------------------------

def prior_windows(L):
    """Offsets of every window the earlier beta pipeline drew at length L,
    as (offset, n_tokens) pairs over val.bin."""
    sizes = beta_common.SPLITS_8192 if L >= 8192 else (
        {"reference": 320, "calibration": 160, "evaluation": 800} if L >= 2048 else None)
    sp = build_splits(L, seed=1234, sizes=sizes)
    return sp, L - 1


def doc_ids():
    data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    return np.cumsum(np.asarray(data) == BOS_TOKEN_ID)


def make_splits(L, bos, n_test, seed=20261001, n_cal=320, n_val=160):
    """Returns dict name -> offsets, plus provenance. Window token span is
    L-1 when a BOS is prepended, else L."""
    data_len = os.path.getsize(VAL_BIN) // 2
    span = L - 1 if bos else L
    used = np.zeros(data_len + 1, dtype=np.int32)
    prior = {}
    for LL in sorted({L, 128, 1024}):
        sp, tok = prior_windows(LL)
        prior[LL] = {k: len(v) for k, v in sp.items()}
        for v in sp.values():
            for off in v:
                used[off] += 1
                used[off + tok] -= 1
    used = np.cumsum(used)[:data_len] > 0
    sp, _ = prior_windows(L)
    cal = sp["reference"][:n_cal]
    val = sp["calibration"][:n_val]
    # fresh windows: a non-overlapping grid over val.bin, keep windows with no used token
    starts = np.arange(0, data_len - span, span)
    csum = np.concatenate([[0], np.cumsum(used)])
    free = (csum[starts + span] - csum[starts]) == 0
    cand = starts[free]
    rng = np.random.default_rng(seed)
    test = np.sort(rng.permutation(cand)[:n_test])
    # document overlap of test with cal/val (reported, not enforced)
    did = doc_ids()
    calval_docs = set()
    for off in np.concatenate([cal, val]):
        calval_docs.update(np.unique(did[off:off + span]).tolist())
    shares = np.array([bool(set(np.unique(did[o:o + span]).tolist()) & calval_docs) for o in test])
    prov = {"L": L, "bos": bos, "span_tokens": span, "seed": seed,
            "calibration": "earlier 'reference' split (seed 1234), first %d" % len(cal),
            "validation": "earlier 'calibration' split (seed 1234), first %d" % len(val),
            "test": "fresh grid windows with no token in any earlier window at L in %s" % sorted(prior),
            "n_test_candidates": int(len(cand)), "n": {"calibration": len(cal), "validation": len(val), "test": len(test)},
            "test_doc_overlap_with_calval_frac": float(shares.mean()) if len(shares) else float("nan"),
            "test_no_doc_overlap_count": int((~shares).sum())}
    if not bos:
        # native LM windows start at the offset itself; earlier LM windows used L-1 tokens
        prov["note"] = "native windows take L raw tokens from the offset"
    return {"calibration": np.asarray(cal), "validation": np.asarray(val), "test": test,
            "test_strict": test[~shares]}, prov


def splits_for(key, n_test=None):
    spec = SPECS[key]
    if n_test is None:
        n_test = 400 if spec.L >= 1024 else 1600
    path = OUT_ROOT / "configs" / f"splits_L{spec.L}_{'bos' if spec.bos else 'raw'}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in ["calibration", "validation", "test", "test_strict"]}, json.loads(str(z["prov"]))
    sp, prov = make_splits(spec.L, spec.bos, n_test)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **sp, prov=json.dumps(prov))
    return sp, prov


def bins_for(L):
    b = {"early": (1, 15), "middle": (16, min(127, L - 1))}
    if L > 128:
        b["late"] = (128, L - 1)
    return b


def jdump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
