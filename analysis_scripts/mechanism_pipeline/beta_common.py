"""Shared code for the BOS-share (beta) mechanism spec (beta_spec.md).

One adapter per model family exposes the same four things:
  pre(idx)          -> Layer-1 and Layer-2 inputs of a clean forward
  attn2(q,k,v)      -> Layer-2 attention recomputed from edited inputs
  finish(h1, o2)    -> decoder features, frozen-head position, LM loss
  decoder features  -> what the frozen head and the refit probe read

"Layer 1" and "Layer 2" are blocks 0 and 1. For the 1H/12H position models
the frozen decoder is the trained pos_head on ln_f(h2). For the 6-layer LM it
is a ridge probe on the Layer-2 residual output (block-1 output), fitted on
clean calibration activations, as in lm_interventions.py.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CHECKPOINTS, NANOGPT, REPO_ROOT, RESULTS_ROOT, BOS_TOKEN_ID, attn_scores_weights,
                    batch_from_offsets, build_splits, forward_capture,
                    head_weights, load_model, ov_images)

LM_CKPT = REPO_ROOT / "nanoGPT/out-lm-6layer-fulltrain-ddp/ckpt.pt"

# model key -> (loader kind, pipeline model name, canonical L, run_id)
MODELS = {
    "1h": ("toy", "attn2_full", 1024, "attn2_full_L1024"),
    "1h_8192": ("toy", "attn2_full_8192", 8192, "attn2_full_8192_L8192"),
    "1h_seed1": ("toy", "attn2_full_seed1", 128, "attn2_full_seed1_L128"),
    "1h_seed2": ("toy", "attn2_full_seed2", 128, "attn2_full_seed2_L128"),
    "1h_seed3": ("toy", "attn2_full_seed3", 128, "attn2_full_seed3_L128"),
    "1h_seed4": ("toy", "attn2_full_seed4", 128, "attn2_full_seed4_L128"),
    "12h": ("toy", "full12h", 1024, "full12h_L1024"),
    "lm": ("lm", None, 128, "lm6_L128"),
}

# length sweep, Layer-1 controls and 1-layer models (beta spec, sweep)
SWEEP_LS = [128, 256, 512, 1024, 2048, 8192]


def _register(key, ckpt_rel, L, run_id):
    CHECKPOINTS[key] = NANOGPT / ckpt_rel
    MODELS[key] = ("toy", key, L, run_id)


for _L in SWEEP_LS:
    for _s in (42, 1, 2):
        _register(f"sw_L{_L}_s{_s}", f"out-sweep-1H-L{_L}-s{_s}/run/best_ckpt.pt", _L, f"sweep/L{_L}_s{_s}")
for _v in ("unifL1", "noMLP1"):
    for _s in (42, 1):
        _register(f"sw_L1024_{_v}_s{_s}", f"out-sweep-1H-L1024-{_v}-s{_s}/run/best_ckpt.pt", 1024, f"sweep/L1024_{_v}_s{_s}")
_register("h1_1layer_s42", "out-mechanism-ATTN1-1024-lr3e4/run/best_ckpt.pt", 1024, "h1_1layer_s42")
_register("h1_1layer_s1", "out-mechanism-ATTN1-1024-lr3e4-s1/run/best_ckpt.pt", 1024, "h1_1layer_s1")

for _s in (42, 1):
    _register(f"sw_L1024_relu_s{_s}", f"out-sweep-1H-L1024-relu-s{_s}/run/best_ckpt.pt", 1024, f"sweep/L1024_relu_s{_s}")
_register("pv_unifL1_s42", "previews/unifL1_s42.pt", 1024, "sweep/preview_L1024_unifL1_s42")

# extra seeds of the paper models (1H, 12H at L = 1024) and the BOS-trained LM
for _s in (1, 2, 3):
    _register(f"seed_12h_s{_s}", f"out-mechanism-12H-1024-seed{_s}/run/best_ckpt.pt", 1024, f"seeds/12h_s{_s}")
_register("seed_1h_s3", "out-mechanism-1H-1024-seed3/run/best_ckpt.pt", 1024, "seeds/1h_s3")
MODELS["lmbos_L1024_s17"] = ("lm", "out-lm-6layer-fulltrain-bos-L1024-seed17/ckpt_20000.pt", 1024, "seeds/lmbos_L1024_s17")
MODELS["lmbos_s42"] = ("lm", "out-lm-6layer-fulltrain-bos/ckpt.pt", 128, "seeds/lmbos_s42")
for _s in range(1, 10):
    MODELS[f"lmbos_s{_s}"] = ("lm", f"out-lm-6layer-fulltrain-bos-seed{_s}/ckpt.pt", 128, f"seeds/lmbos_s{_s}")

SPLITS_8192 = {"reference": 128, "calibration": 64, "evaluation": 340}


def spec_splits(L):
    sizes = SPLITS_8192 if L >= 8192 else ({"reference": 320, "calibration": 160, "evaluation": 800}
                                           if L >= 2048 else None)
    return build_splits(L, seed=1234, sizes=sizes)


def bins_for(L):
    b = {"early": (1, 15), "middle": (16, min(127, L - 1))}
    if L > 128:
        b["late"] = (128, L - 1)
    return b


def ln(x, mod):
    return F.layer_norm(x, mod.weight.shape, mod.weight, mod.bias, 1e-5)


class Ridge:
    """Ridge regression by normal equations, fp64 accumulation."""

    def __init__(self, dim, device, lam=1e-4):
        self.A = torch.zeros(dim + 1, dim + 1, device=device, dtype=torch.float64)
        self.b = torch.zeros(dim + 1, device=device, dtype=torch.float64)
        self.lam = lam

    def add(self, X, y):
        X = X.double().reshape(-1, X.shape[-1])
        X1 = torch.cat([X, torch.ones(len(X), 1, device=X.device,
                                      dtype=torch.float64)], 1)
        self.A += X1.T @ X1
        self.b += X1.T @ y.double().reshape(-1)

    def solve(self):
        n = self.A.shape[0]
        reg = self.lam * torch.diag(self.A).mean() * torch.eye(
            n, device=self.A.device, dtype=torch.float64)
        reg[-1, -1] = 0
        self.w = torch.linalg.solve(self.A + reg, self.b)
        return self

    def predict(self, X):
        X1 = torch.cat([X.double(), torch.ones(*X.shape[:-1], 1, device=X.device,
                                               dtype=torch.float64)], -1)
        return X1 @ self.w


@torch.no_grad()
def attn_from_inputs(attn, q_in, k_in, v_in, return_head_out=False):
    """Recompute one attention module from separate q/k/v inputs.

    Each input is [B,T,d] (shared by all heads) or [B,H,T,d] (per head).
    Returns (weights [B,H,T,T], o [B,T,d]) where o includes all biases and
    c_proj exactly as the model does. fp32.
    """
    hw = head_weights(attn)

    def proj(x, W, b):
        if x.dim() == 3:
            return torch.einsum("btd,hed->bhte", x.float(), W.float()) + b.float()[None, :, None, :]
        return torch.einsum("bhtd,hed->bhte", x.float(), W.float()) + b.float()[None, :, None, :]

    q = proj(q_in, hw.W_q, hw.b_q)
    k = proj(k_in, hw.W_k, hw.b_k)
    v = proj(v_in, hw.W_v, hw.b_v)
    s = torch.einsum("bhte,bhse->bhts", q, k) * hw.scale
    T = s.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=s.device, dtype=torch.bool), 1)
    w = torch.softmax(s.masked_fill(mask, float("-inf")), -1)
    hv = torch.einsum("bhts,bhse->bhte", w, v)                 # [B,H,T,dh]
    o = torch.einsum("bhte,hde->btd", hv, hw.W_o.float()) + hw.b_o.float()
    if return_head_out:
        return w, o, hv
    return w, o


class ToyAdapter:
    """2-layer position models (1H, 12H)."""

    kind = "toy"

    def __init__(self, name, device, init_only=False):
        self.model, self.meta = load_model(name, device, init_only=init_only)
        m = self.model
        self.device = device
        self.H, self.d = m.config.n_head, m.config.n_embd
        self.L_train = self.meta["block_size_train"]
        self.one_layer = m.config.n_layer == 1
        self.attn1, self.attn2 = m.block1.attn, m.block2.attn
        self.ln1_bias = m.block1.ln_1.bias
        self.ln2_bias = m.block2.ln_1.bias
        if self.one_layer:
            # 1-layer control: its only layer plays the "last layer" role, so
            # every Layer-2 analysis reads block 1 with input x1.
            self.attn2 = m.block1.attn
            self.ln2_bias = m.block1.ln_1.bias
        self.probe = None

    @torch.no_grad()
    def pre(self, idx):
        if self.one_layer:
            m = self.model
            out, _ = m(idx, targets=None, capture_taps=True)
            taps = m.get_all_taps()
            a1 = m.block1.attn.last_attention_weights
            x1, emb = taps["block1_ln1"], taps["embeddings"]
            s1, _ = attn_scores_weights(x1, head_weights(m.block1.attn))
            o1 = taps["block1_post_attn"] - emb
            return {"x1": x1, "attn1_w": a1, "post1": taps["block1_post_attn"],
                    "h1bar": taps["block1_ln2"], "h1": emb, "x2": x1,
                    "attn2_w": a1, "scores2": s1, "o2": o1, "o1": o1,
                    "pred": out.squeeze(-1) * (self.L_train - 1)}
        cap = forward_capture(self.model, idx)
        return {"x1": cap["x1"], "attn1_w": cap["attn1_w"], "post1": cap["post1"],
                "h1bar": cap["h1bar"], "h1": cap["h1"], "x2": cap["x2"],
                "attn2_w": cap["attn2_w"], "scores2": cap["scores2"],
                "o2": cap["o2"], "o1": cap["post1"] - cap["emb"],
                "pred": cap["pred"] * (self.L_train - 1)}

    @torch.no_grad()
    def finish(self, h1, o2, idx=None):
        m = self.model
        if self.one_layer:                    # h1 = embeddings, o2 = the layer's attention update
            post = h1 + o2.float()
            h2 = post + m.block1.mlp(ln(post, m.block1.ln_2))
        else:
            post2 = h1 + o2.float()
            h2 = post2 + m.block2.mlp(ln(post2, m.block2.ln_2))
        feats = m.ln_f(h2)
        frozen = m.pos_head(feats).squeeze(-1).double() * (self.L_train - 1)
        return feats, frozen, None

    def fit_frozen(self, *_):
        return None


class LMAdapter:
    """6-layer NoPE LM; Layer 2 = block index 1."""

    kind = "lm"

    def __init__(self, device, ckpt=LM_CKPT):
        from lm_mechanism import load_lm
        self.model, self.cfg, _ = load_lm(str(ckpt), device)
        self.device = device
        self.blocks = self.model.transformer.h
        self.H, self.d = self.cfg.n_head, self.cfg.n_embd
        self.attn1, self.attn2 = self.blocks[0].attn, self.blocks[1].attn
        self.ln1_bias = self.blocks[0].ln_1.bias
        self.ln2_bias = self.blocks[1].ln_1.bias
        self.meta = {"model_name": "lm6", "checkpoint_path": str(ckpt)}
        self.probe = None

    @torch.no_grad()
    def _rest(self, blk, x, o):
        xa = x + o
        if getattr(blk, "use_ln2", True):
            return xa + blk.mlp(ln(xa, blk.ln_2)), xa
        return xa + blk.mlp(xa), xa

    @torch.no_grad()
    def pre(self, idx):
        b0, b1 = self.blocks[0], self.blocks[1]
        emb = self.model.transformer.wte(idx)
        x1 = ln(emb, b0.ln_1)
        w1, o1 = attn_from_inputs(b0.attn, x1, x1, x1)
        h1, post1 = self._rest(b0, emb, o1)
        x2 = ln(h1, b1.ln_1)
        hw2 = head_weights(b1.attn)
        s2, w2 = attn_scores_weights(x2, hw2)
        o2 = torch.einsum("bhts,bhsd->btd", w2, ov_images(x2, hw2))
        o2 = o2 + torch.einsum("he,hde->d", hw2.b_v.float(), hw2.W_o.float())
        if b1.attn.c_proj.bias is not None:
            o2 = o2 + b1.attn.c_proj.bias.float()
        return {"x1": x1, "attn1_w": w1, "post1": post1, "h1": h1, "x2": x2,
                "attn2_w": w2, "scores2": s2, "o2": o2, "o1": o1,
                "h1bar": ln(post1, b0.ln_2) if getattr(b0, "use_ln2", True) else post1}

    @torch.no_grad()
    def finish(self, h1, o2, idx=None):
        x, _ = self._rest(self.blocks[1], h1, o2.float())
        feats = x
        frozen = self.probe.predict(feats) if self.probe is not None else None
        loss = None
        if idx is not None:
            for blk in self.blocks[2:]:
                x = blk(x)
            logits = self.model.lm_head(self.model.transformer.ln_f(x))
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                idx[:, 1:].reshape(-1), reduction="none").reshape(idx.shape[0], -1)
        return feats, frozen, loss

    @torch.no_grad()
    def fit_frozen(self, offsets, L, batch_size):
        """The LM's frozen decoder: ridge on clean Layer-2 residual output."""
        r = Ridge(self.d, self.device)
        tgt = torch.arange(L, device=self.device, dtype=torch.float64)
        for s in range(0, len(offsets), batch_size):
            idx = batch_from_offsets(offsets[s:s + batch_size], L, self.device)
            p = self.pre(idx)
            feats, _, _ = self.finish(p["h1"], p["o2"])
            r.add(feats, tgt.expand(idx.shape[0], L))
        self.probe = r.solve()
        return self.probe


def make_adapter(key, device, init_only=False):
    kind, name, L, run_id = MODELS[key]
    if kind == "toy":
        return ToyAdapter(name, device, init_only=init_only), L, run_id
    return LMAdapter(device, ckpt=(NANOGPT / name) if name else LM_CKPT), L, run_id


def out_dir(run_id, sub="beta"):
    d = RESULTS_ROOT / run_id / sub
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def gamma_fit(curve, lo, hi):
    """OLS slope of log(curve[i]) on log(i) over i in [lo, hi]."""
    hi = min(hi, len(curve) - 1)
    i = np.arange(lo, hi + 1)
    y = np.asarray(curve, dtype=np.float64)[lo:hi + 1]
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(i[m]), np.log(y[m]), 1)[0])


def gamma_windows(L):
    w = {"16_64": (16, 64), "64_256": (64, 256), "256_1024": (256, 1024),
         "1024_8192": (1024, 8192), "16_127": (16, 127)}
    return {k: v for k, v in w.items() if v[0] < L - 1}


def smooth_deriv(f, L, deg=6, i_min=4):
    """Fit f(i) by a degree-6 polynomial in log i over i >= i_min and
    return (fitted f, df/di) on i = 0..L-1 (nan below i_min)."""
    i = np.arange(L, dtype=np.float64)
    m = (i >= i_min) & np.isfinite(f)
    u = np.log(i[m])
    c = np.polyfit(u, f[m], deg)
    fit = np.full(L, np.nan)
    der = np.full(L, np.nan)
    uu = np.log(i[i >= i_min])
    fit[i >= i_min] = np.polyval(c, uu)
    der[i >= i_min] = np.polyval(np.polyder(c), uu) / i[i >= i_min]
    return fit, der


def err_by_bin(pred, L, bins):
    """pred [n,T] absolute positions. relMAE = median over (seq,pos) in the
    bin of |yhat-i|/i; MAE = mean |yhat-i|."""
    i = np.arange(L)[None, :].repeat(len(pred), 0)
    out = {}
    for name, (a, b) in bins.items():
        sel = (i >= a) & (i <= b)
        e = np.abs(pred[sel] - i[sel])
        out[f"relmae_{name}"] = float(np.median(e / i[sel]))
        out[f"mae_{name}"] = float(e.mean())
    sel = i >= 1
    out["mae_all"] = float(np.abs(pred[sel] - i[sel]).mean())
    # R^2 over positions 1..L-1, the same positions as the other metrics.
    y = i[sel].astype(np.float64)
    out["r2_all"] = float(1 - np.sum((pred[sel] - y) ** 2) / np.sum((y - y.mean()) ** 2))
    return out
