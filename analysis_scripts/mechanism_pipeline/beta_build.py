"""A fully hand-set 2-layer NoPE position model (the construction).

Architecture: the 1H model (d = 768, one head, pre-norm LayerNorm, ReLU MLPs).
Every weight is set by hand; nothing is trained except the decoders.

  embeddings  random N(0, 1) per coordinate (never trained)
  Layer 1     attention Q = K = 0, so the attention is exactly uniform.
              Value path: read "how BOS-like is the token" (the component of
              its input along the BOS input direction) and write it, times
              `gain`, along one fixed direction e. So after attention, token j
              holds about gain * 27.7 / (j + 1) along e (BOS amount) plus a
              small noise term from the other tokens.
  Layer-1 MLP two ReLU units read +e and -e after the pre-MLP LayerNorm, so
              together they compute the amount along e divided by the token's
              LayerNorm scale. The MLP writes minus that amount times the
              typical scale of an ordinary token, so ordinary tokens are
              erased exactly. BOS-heavy tokens have a larger scale, so the
              erasure falls short: they stay BOS-like.
  Layer 2     score of key j = B x (BOS amount of key j at the Layer-2 input);
              the value writes the same amount along an output direction w.
              So the attention output is (attention-weighted BOS amount) x w.
  Layer-2 MLP zero.

Two constants are read from reference data, not trained: the BOS input of
Layer 2 and the mean Layer-2 input (to measure the BOS amount of each key).

Controls: `no_mlp1` (Layer-1 MLP off) and `one_layer` (Layer 2 reads the
LayerNorm'd embeddings directly, no Layer 1).
Decoders: isotonic regression from the output amount to position (fit on the
calibration split), and a linear head on the final LayerNorm output.

Usage: python beta_build.py --device cuda:0 [--save-ckpt]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beta_analyze import isotonic_decode
from beta_common import RESULTS_ROOT, Ridge, bins_for, err_by_bin, ln, spec_splits
from beta_v2 import free_fit, n_ordinary
from common import NANOGPT, add_nanogpt_to_path, batch_from_offsets

L = 1024
D = 768
N_CAL, N_EVAL = 320, 800
B_SWEEP = [3.0, 4.0, 5.0, 6.0, 7.0]
GAIN_SWEEP = [2.0, 4.0, 8.0]
W_NORM = 50.0


def unit_zero_mean(x):
    x = x - x.mean()
    return x / x.norm()


@torch.no_grad()
def build(gain, B, variant, dev, seed=0):
    """Return a hand-set TwoLayerMechanismModel and a dict of its constants."""
    add_nanogpt_to_path()
    from model_2layer_mechanism import TwoLayerMechanismConfig, TwoLayerMechanismModel
    torch.manual_seed(seed)
    cfg = TwoLayerMechanismConfig(block_size=L, n_head=1, n_embd=D, bias=True,
                                  activation="relu", use_regression=True, use_flash=False)
    m = TwoLayerMechanismModel(cfg).to(dev)
    for p_ in m.parameters():
        p_.zero_()
    g = torch.Generator(device="cpu").manual_seed(seed)
    m.wte.weight.copy_(torch.randn(m.wte.weight.shape, generator=g))
    e = unit_zero_mean(torch.randn(D, generator=g)).to(dev)          # Layer-1 BOS direction
    w = unit_zero_mean(torch.randn(D, generator=g)).to(dev) * W_NORM  # Layer-2 output direction
    for mod in [m.block1.ln_1, m.block1.ln_2, m.block2.ln_1, m.block2.ln_2, m.ln_f]:
        mod.weight.fill_(1.0)
    bos_id = 50256
    x_bos1 = ln(m.wte.weight[bos_id][None, None], m.block1.ln_1)[0, 0]  # Layer-1 input of BOS
    b_hat = x_bos1 / x_bos1.norm()
    # Layer-1 attention: Q = K = 0 (uniform); V reads <x, b_hat>, O writes gain * e
    a1 = m.block1.attn
    a1.c_attn.weight[2 * D] = b_hat                                   # value row 0
    a1.c_proj.weight[:, 0] = gain * e
    C = {"gain": gain, "B": B, "variant": variant}
    return m, e, w, C


@torch.no_grad()
def calibrate(m, e, w, B, variant, offsets, dev, bs=8):
    """Set the Layer-1 MLP (needs the ordinary-token scale) and Layer 2
    (needs the BOS input and mean input of Layer 2) from reference data."""
    def post1_of(idx):
        emb = m.wte(idx)
        x1 = m.block1.ln_1(emb)
        return emb + m.block1.attn(x1)

    sig = []
    for s in range(0, len(offsets), bs):
        idx = batch_from_offsets(offsets[s:s + bs], L, dev)
        p1 = post1_of(idx)
        sig.append(p1.std(-1, unbiased=False)[:, 128:].reshape(-1))
    sigma_tail = float(torch.cat(sig).mean())
    mlp = m.block1.mlp
    if variant != "no_mlp1":
        mlp.c_fc.weight[0] = e; mlp.c_fc.weight[1] = -e              # ReLU(z) - ReLU(-z) = z
        mlp.c_proj.weight[:, 0] = -sigma_tail * e
        mlp.c_proj.weight[:, 1] = sigma_tail * e

    def x2_of(idx):
        if variant == "one_layer":
            return m.block1.ln_1(m.wte(idx))
        p1 = post1_of(idx)
        h1 = p1 + mlp(m.block1.ln_2(p1))
        return m.block2.ln_1(h1)

    xs = torch.zeros(D, device=dev, dtype=torch.float64); n = 0; xb = None
    for s in range(0, len(offsets), bs):
        idx = batch_from_offsets(offsets[s:s + bs], L, dev)
        x2 = x2_of(idx).double()
        xb = x2[0, 0].clone() if xb is None else xb
        xs += x2[:, 1:].sum((0, 1)); n += idx.shape[0] * (L - 1)
    mu = xs / n
    v = xb - mu
    k = v / (v @ v)                                                   # <x - mu, k> = BOS amount
    a2 = m.block2.attn
    a2.c_attn.weight[D] = k.float(); a2.c_attn.bias[D] = -(mu @ k).float()          # key row 0
    a2.c_attn.bias[0] = float(np.sqrt(D) * B)                                      # query = sqrt(d) B e_1
    a2.c_attn.weight[2 * D] = k.float(); a2.c_attn.bias[2 * D] = -(mu @ k).float()  # value = BOS amount
    a2.c_proj.weight[:, 0] = w
    return {"sigma_tail": sigma_tail, "x2_of": x2_of}


@torch.no_grad()
def forward(m, x2_of, idx, variant):
    """Hand-set forward: returns BOS amounts of the keys, BOS attention,
    output amount t (along w), and final features for the linear head."""
    x2 = x2_of(idx)
    a2 = m.block2.attn
    o2, att = a2(x2, return_attn_weights=True)
    k = a2.c_attn.weight[D].double(); kb = a2.c_attn.bias[D].double()
    amount = x2.double() @ k + kb                                     # BOS amount of each key
    t = torch.einsum("bij,bj->bi", att[:, 0].double(), amount)
    if variant == "one_layer":
        h1 = m.wte(idx)
    else:
        emb = m.wte(idx); p1 = emb + m.block1.attn(m.block1.ln_1(emb))
        h1 = p1 + m.block1.mlp(m.block1.ln_2(p1))
    feats = m.ln_f(h1 + o2)
    return amount, att[:, 0, :, 0].double(), t, feats


def evaluate(gain, B, variant, dev):
    splits = spec_splits(L)
    m, e, w, C = build(gain, B, variant, dev)
    cal = calibrate(m, e, w, B, variant, splits["reference"], dev)
    bins = bins_for(L)
    tgt = torch.arange(L, device=dev, dtype=torch.float64)
    ridge = Ridge(D, dev)
    t_cal = []
    for s in range(0, N_CAL, 8):
        idx = batch_from_offsets(splits["calibration"][s:s + 8], L, dev)
        _, _, t, feats = forward(m, cal["x2_of"], idx, variant)
        ridge.add(feats, tgt.expand(idx.shape[0], L)); t_cal.append(t.cpu())
    ridge.solve()
    amt, ab, tt, lin = [], [], [], []
    for s in range(0, N_EVAL, 8):
        idx = batch_from_offsets(splits["evaluation"][s:s + 8], L, dev)
        amount, abos, t, feats = forward(m, cal["x2_of"], idx, variant)
        amt.append(amount.cpu()); ab.append(abos.cpu()); tt.append(t.cpu()); lin.append(ridge.predict(feats).cpu())
    amt, ab, tt, lin = (torch.cat(x).numpy() for x in (amt, ab, tt, lin))
    iso = isotonic_decode(torch.cat(t_cal).numpy(), tt, L)
    # count of BOS-like keys and the counting formula, measured on this model
    frac = (amt[:, 1:] > 0.5).mean(0)
    sinks = [int(j) + 1 for j in np.nonzero(frac > 0.9)[0]]
    nb = 1 + len(sinks)
    ordinary = np.ones(L, bool); ordinary[0] = False; ordinary[sinks] = False; ordinary[:16] = False
    amt_ord = float(amt[:, ordinary].mean())
    nord = n_ordinary(L, sinks)
    pref = 1 + np.array([sum(1 for s_ in sinks if s_ <= j) for j in range(L)])
    pred = np.exp(B) / (pref * np.exp(B) + nord * np.exp(B * amt_ord))
    meas = ab.mean(0)
    rel = np.abs(pred - meas) / meas
    return {"gain": gain, "B": B, "variant": variant, "sigma_tail": cal["sigma_tail"],
            "bos_amount_profile": {str(j): float(amt[:, j].mean()) for j in [1, 2, 3, 4, 8, 16, 64, 256, 1023]},
            "bos_amount_sd_ordinary": float(amt[:, ordinary].std()),
            "n_bos_like_keys": nb, "formula_relerr_by_bin": {k: float(np.median(rel[a:b + 1])) for k, (a, b) in bins.items()},
            "bos_weight_cv_across_seq_at": {str(j): float(ab[:, j].std() / ab[:, j].mean()) for j in [16, 128, 1023]},
            "isotonic": err_by_bin(iso, L, bins), "linear_head": err_by_bin(lin, L, bins)}


@torch.no_grad()
def save_construction(gain, B, dev):
    """Save a calibrated full construction as a training start point. The
    Layer-2 MLP gets the standard random init (a zero ReLU MLP cannot learn)."""
    splits = spec_splits(L)
    m, e, w, _ = build(gain, B, "full", dev)
    calibrate(m, e, w, B, "full", splits["reference"], dev)
    for lin in [m.block2.mlp.c_fc, m.block2.mlp.c_proj]:
        torch.nn.init.normal_(lin.weight, mean=0.0, std=0.02)
    out = NANOGPT / "constructions"; out.mkdir(exist_ok=True)
    path = out / f"full_g{gain:g}_B{B:g}.pt"
    torch.save({"model": m.state_dict(), "gain": gain, "B": B}, path)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save", default="", help="comma list gain:B to save as start points, then exit")
    args = ap.parse_args()
    if args.save:
        for item in args.save.split(","):
            gain, B = map(float, item.split(":"))
            save_construction(gain, B, args.device)
        return
    od = RESULTS_ROOT / "construction"; od.mkdir(parents=True, exist_ok=True)
    res = []
    for variant in ["full", "no_mlp1", "one_layer"]:
        for gain in (GAIN_SWEEP if variant != "one_layer" else [GAIN_SWEEP[0]]):
            for B in B_SWEEP:
                r = evaluate(gain, B, variant, args.device)
                res.append(r)
                print(f"{variant:9s} gain={gain:g} B={B:g} keys={r['n_bos_like_keys']} "
                      f"amount j1..4={[round(r['bos_amount_profile'][str(j)], 2) for j in (1, 2, 3, 4)]} "
                      f"ord sd={r['bos_amount_sd_ordinary']:.4f} formula err late={r['formula_relerr_by_bin']['late']:.3f} "
                      f"iso relMAE e/m/l={r['isotonic']['relmae_early']:.3f}/{r['isotonic']['relmae_middle']:.4f}/{r['isotonic']['relmae_late']:.4f} "
                      f"lin late={r['linear_head']['relmae_late']:.4f}", flush=True)
    (od / "build_sweep.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
