"""
MLP Probe Analysis for OWT Experiments
"""

import sys
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanoGPT"))
from model_nope import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results/mlp_probe_owt")
PLOTS_DIR = Path("overleaf/nopos---claude-version/plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "nope_ln": {
        "checkpoint": "nanoGPT/out-nope-owt-ln/ckpt_13000.pt",
        "name": "NoPE+LN",
        "args": {
            "use_positional_embedding": False,
            "norm_type": "layernorm",
            "skip_ln2": False,
            "use_batchnorm_ln2": False,
        },
    },
    "nope_bn2": {
        "checkpoint": "nanoGPT/out-nope-owt-bn2/ckpt_18000.pt",
        "name": "NoPE+BN2",
        "args": {
            "use_positional_embedding": False,
            "norm_type": "layernorm",
            "skip_ln2": False,
            "use_batchnorm_ln2": True,
        },
    },
    "nope_no_ln2": {
        "checkpoint": "nanoGPT/out-nope-owt-no-ln2/ckpt_18000.pt",
        "name": "NoPE+NoLN2",
        "args": {
            "use_positional_embedding": False,
            "norm_type": "layernorm",
            "skip_ln2": True,
            "use_batchnorm_ln2": False,
        },
    },
    "baseline_pe": {
        "checkpoint": "nanoGPT/out-baseline-owt-pe/ckpt_13000.pt",
        "name": "Baseline+PE",
        "args": {
            "use_positional_embedding": True,
            "norm_type": "layernorm",
            "skip_ln2": False,
            "use_batchnorm_ln2": False,
        },
    },
}

BASE_ACTIVATION_POINTS = ["raw_embed", "post_ln1", "post_attn", "post_mlp_residual"]


def create_model(args, seq_len):
    cfg = GPTConfig(
        n_layer=1,
        n_head=12,
        n_embd=768,
        block_size=seq_len,
        vocab_size=50257,
        bias=False,
        dropout=0.0,
        **args,
    )
    return GPT(cfg).to(device).eval()


def load_trained(path, args):
    ckpt = torch.load(path, map_location=device)
    model_args = ckpt["model_args"].copy()
    model_args.update(args)
    cfg = GPTConfig(**model_args)
    model = GPT(cfg).to(device).eval()
    state = ckpt["model"]
    for k in list(state.keys()):
        if k.startswith("_orig_mod."):
            state[k[10:]] = state.pop(k)
    model.load_state_dict(state)
    return model


def get_activations(model, ids):
    acts = {}
    with torch.no_grad():
        emb = model.transformer.wte(ids)
        acts["raw_embed"] = emb.cpu().numpy()
        x = emb
        block = model.transformer.h[0]
        x_ln1 = block.ln_1(x)
        acts["post_ln1"] = x_ln1.cpu().numpy()
        attn_out = block.attn(x_ln1)
        x_attn = x + attn_out
        acts["post_attn"] = x_attn.cpu().numpy()
        has_ln2 = hasattr(block, "ln_2")
        if has_ln2:
            x_ln2 = block.ln_2(x_attn)
            acts["post_ln2"] = x_ln2.cpu().numpy()
            mlp_in = x_ln2
        else:
            mlp_in = x_attn
        mlp_out = block.mlp(mlp_in)
        x_mlp = mlp_in + mlp_out
        acts["post_mlp_residual"] = x_mlp.cpu().numpy()
    return acts, has_ln2


def run_experiment(model, n_train, n_test, seq_len, seed):
    torch.manual_seed(seed)
    train_ids = torch.randint(0, 50257, (n_train, seq_len), device=device)
    test_ids = torch.randint(0, 50257, (n_test, seq_len), device=device)

    train_acts = {}
    test_acts = {}

    batch = 64
    has_ln2 = None
    for i in range(0, n_train, batch):
        acts, has_ln2 = get_activations(model, train_ids[i : i + batch])
        for k, v in acts.items():
            train_acts[k] = train_acts.get(k, []) + [v]
    for i in range(0, n_test, batch):
        acts, _ = get_activations(model, test_ids[i : i + batch])
        for k, v in acts.items():
            test_acts[k] = test_acts.get(k, []) + [v]

    for k in train_acts:
        train_acts[k] = np.concatenate(train_acts[k])
        test_acts[k] = np.concatenate(test_acts[k])

    train_pos = np.tile(np.arange(seq_len), n_train)
    test_pos = np.tile(np.arange(seq_len), n_test)

    results = {}
    for act in sorted(train_acts.keys()):
        d = train_acts[act].shape[-1]
        X_train = train_acts[act].reshape(-1, d)
        X_test = test_acts[act].reshape(-1, d)

        lin = Ridge(alpha=1.0).fit(X_train, train_pos)
        lin_r2 = r2_score(test_pos, lin.predict(X_test))

        mlp = MLPRegressor(
            hidden_layer_sizes=(32,), max_iter=30, early_stopping=True, random_state=42
        )
        mlp.fit(X_train, train_pos)
        mlp_r2 = r2_score(test_pos, mlp.predict(X_test))

        results[f"{act}_linear"] = {"r2": lin_r2}
        results[f"{act}_mlp"] = {"r2": mlp_r2}
        print(f"  {act}: Linear={lin_r2:.3f}, MLP={mlp_r2:.3f}")

    results["_has_ln2"] = has_ln2
    return results


def plot_figure(all_results, save_path, title, bar_mode=False):
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            "NoPE+LN (Rand)",
            "NoPE+LN (Train)",
            "NoPE+BN2 (Rand)",
            "NoPE+BN2 (Train)",
            "NoPE+NoLN2 (Rand)",
            "NoPE+NoLN2 (Train)",
            "Baseline+PE (Rand)",
            "Baseline+PE (Train)",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.05,
    )

    for exp_idx, exp_key in enumerate(EXPERIMENTS.keys()):
        for model_idx, model_type in enumerate(["random", "trained"]):
            col = exp_idx * 2 + model_idx + 1
            r = all_results.get(f"{exp_key}_{model_type}", {})

            acts = sorted(
                [
                    k.replace("_linear", "")
                    for k in r.keys()
                    if k.endswith("_linear") and not k.startswith("_")
                ]
            )
            if not acts:
                continue
            lin = [r.get(f"{a}_linear", {}).get("r2", 0) for a in acts]
            mlp = [r.get(f"{a}_mlp", {}).get("r2", 0) for a in acts]
            x = list(range(len(acts)))

            if not bar_mode:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=lin,
                        mode="lines+markers",
                        name="Linear",
                        line=dict(color="#1f77b4", width=2),
                        marker=dict(size=6),
                        showlegend=(exp_idx == 0 and model_idx == 0),
                    ),
                    row=1,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mlp,
                        mode="lines+markers",
                        name="MLP",
                        line=dict(color="#ff7f0e", width=2, dash="dash"),
                        marker=dict(size=6, symbol="diamond"),
                        showlegend=(exp_idx == 0 and model_idx == 0),
                    ),
                    row=1,
                    col=col,
                )
            else:
                gain = [mlp[i] - lin[i] for i in range(len(lin))]
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=gain,
                        marker_color=["#2ca02c" if g > 0 else "#d62728" for g in gain],
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )

            fig.update_xaxes(
                tickvals=list(range(len(acts))),
                ticktext=[a.replace("post_", "").replace("_", "\n") for a in acts],
                tickangle=45,
                row=1,
                col=col,
            )
            fig.update_yaxes(
                range=[-0.1, 1.0] if not bar_mode else None,
                title_text="R²" if col in [1, 3, 5, 7] else None,
                row=1,
                col=col,
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Serif"), x=0.5),
        width=1400,
        height=600,
        template="plotly_white",
        legend=dict(x=0.35, y=1.02, orientation="h"),
    )
    fig.write_image(f"{save_path}.png", width=1400, height=600, scale=2)
    fig.write_image(f"{save_path}.pdf")
    print(f"Saved {save_path}")


def main():
    n_samples, seq_len, seed = 400, 64, 42
    n_train, n_test = int(n_samples * 0.75), int(n_samples * 0.25)

    print(f"MLP Probe: {n_samples} samples, seq_len={seq_len}")
    all_results = {}

    for exp_key, exp_cfg in EXPERIMENTS.items():
        print(f"\n## {exp_cfg['name']} ##")
        rand = create_model(exp_cfg["args"], seq_len)
        print("Random:")
        all_results[f"{exp_key}_random"] = run_experiment(
            rand, n_train, n_test, seq_len, seed
        )
        del rand
        torch.cuda.empty_cache()

        if Path(exp_cfg["checkpoint"]).exists():
            trained = load_trained(exp_cfg["checkpoint"], exp_cfg["args"])
            print("Trained:")
            all_results[f"{exp_key}_trained"] = run_experiment(
                trained, n_train, n_test, seq_len, seed
            )
            del trained
            torch.cuda.empty_cache()

    print("\nGenerating figures...")
    plot_figure(
        all_results,
        str(RESULTS_DIR / "mlp_probes_fig1"),
        "Figure 1: MLP vs Linear Probe Position Decoding",
    )
    plot_figure(
        all_results,
        str(PLOTS_DIR / "mlp_probes_fig1"),
        "Figure 1: MLP vs Linear Probe Position Decoding",
    )
    plot_figure(
        all_results,
        str(RESULTS_DIR / "mlp_probes_fig2"),
        "Figure 2: MLP Advantage (MLP R² - Linear R²)",
        bar_mode=True,
    )
    plot_figure(
        all_results,
        str(PLOTS_DIR / "mlp_probes_fig2"),
        "Figure 2: MLP Advantage (MLP R² - Linear R²)",
        bar_mode=True,
    )

    with open(RESULTS_DIR / "mlp_probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone! Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
