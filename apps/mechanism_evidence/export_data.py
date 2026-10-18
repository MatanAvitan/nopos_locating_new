"""Read the selected cached evidence without loading models or running experiments."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


# Keep the selected checkpoints fixed across all four steps.
MODEL_SELECTION = {
    "1h": {
        "name": "1H", "kind": "toy", "seed": 42, "iteration": 19500, "heads": [0], "recorded_heads": [0],
        "trained_context_length": 1024,
        "checkpoint_path": "nanoGPT/out-mechanism-ATTN2FULL-1024-lr3e4/R0/ppe56vt4/best_ckpt.pt",
        "metadata_path": "results/mechanism/attn2_full_L1024/beta/meta.json",
    },
    "seed_12h_s1": {
        "name": "12H", "kind": "toy", "seed": 1, "iteration": 17500,
        "heads": [2, 3, 5, 9, 10, 11], "recorded_heads": [2, 3, 5, 9, 10, 11],
        "trained_context_length": 1024,
        "checkpoint_path": "nanoGPT/out-mechanism-12H-1024-seed1/run/best_ckpt.pt",
        "metadata_path": "results/mechanism/seeds/12h_s1/beta/meta.json",
    },
    "lmbos_s9": {
        "name": "LM-BOS L128", "kind": "lm", "seed": 9, "iteration": 20000,
        "heads": [0], "recorded_heads": [0, 3, 5, 9], "trained_context_length": 128,
        "context_source": "logs/lm_bos_extra_20261005/checkpoint_metadata.json",
        "checkpoint_path": "nanoGPT/out-lm-6layer-fulltrain-bos-seed9/ckpt.pt",
        "metadata_path": "results/mechanism/seeds/lmbos_s9/beta/meta.json",
    },
    # Trained at L = 1024. The main head is h7, the layer-2 head with the largest BOS ratio r_h (622.5).
    "lmbos_L1024_s2": {
        "name": "LM-BOS L1024", "kind": "lm", "seed": 2, "iteration": 20000,
        "heads": [7], "recorded_heads": [0, 1, 2, 3, 5, 6, 7, 10], "trained_context_length": 1024,
        "context_source": "logs/lm_bos_L1024_eval/checkpoint_metadata_s2.json",
        "checkpoint_path": "nanoGPT/out-lm-6layer-fulltrain-bos-L1024-seed2/ckpt_20000.pt",
        "metadata_path": "results/mechanism/seeds/lmbos_L1024_s2/beta/meta.json",
    },
}
DIAGNOSTICS = {
    "evidence": "evidence.json",
    "softmax": "softmax_beta_check.json",
    "key_shift": "causal_story.json",
    "interventions": "denoise_interventions_v2.json",
}
COUNT_SCRIPTS = {
    "story": ("story_evidence.py", 400),
    "softmax": ("softmax_beta_check.py", 400),
    "key_shift": ("causal_story.py", 200),
    "interventions": ("denoise_interventions_v2.py", 1600),
}
ARMS_BY_STEP = {
    "2": ("mlp_mean", "mlp_mean_rand"),
    "3": ("keys_noshare", "keys_noshare_rand"),
    "4": ("values_noshare", "values_noshare_rand"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _field(document: dict, path: str):
    """Require a field and identify its complete path in validation errors."""
    value = document
    parts = path[1:].split("/") if path.startswith("/") else path.split(".")
    for part in parts:
        _require(isinstance(value, dict) and part in value, f"Missing field: {path}")
        value = value[part]
    return value


def _number(document: dict, path: str) -> float:
    value = _field(document, path)
    _require(type(value) in (int, float) and math.isfinite(value), f"Invalid finite metric: {path}")
    return value


def _integer(document: dict, path: str, minimum: int = 0) -> int:
    value = _field(document, path)
    _require(type(value) is int and value >= minimum, f"Invalid integer: {path}")
    return value


def _source(path: Path, display_path: str, script: bool = False) -> dict:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        document = text if script else json.loads(text)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Cannot read source: {display_path}") from error
    return {"path": display_path, "sha256": hashlib.sha256(raw).hexdigest(), "document": document}


def _ref(source: str, field: str) -> dict:
    pointer = field if field.startswith("/") else "/" + field.replace(".", "/")
    return {"source": source, "field": pointer}


def _eval_constant(text: str, path: str) -> tuple[int, int]:
    """Read the N_EVAL constant through AST, without executing the script."""
    try:
        tree = ast.parse(text, filename=path)
        assignments = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "N_EVAL":
                        assignments.append((ast.literal_eval(node.value), node.lineno))
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        names = [item.id if isinstance(item, ast.Name) else None for item in target.elts]
                        if "N_EVAL" in names:
                            values = ast.literal_eval(node.value)
                            _require(isinstance(values, (tuple, list)) and len(values) == len(names),
                                     f"Invalid count assignment: {path}")
                            assignments.append((values[names.index("N_EVAL")], node.lineno))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "N_EVAL":
                    assignments.append((ast.literal_eval(node.value), node.lineno))
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                _require(node.target.id != "N_EVAL", f"Modified count constant: {path}")
        _require(len(assignments) == 1, f"Require one N_EVAL assignment: {path}")
        value, line = assignments[0]
        _require(type(value) is int and value > 0, f"Invalid N_EVAL constant: {path}")
        return value, line
    except (SyntaxError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f"Cannot verify N_EVAL: {path}") from error


def _counts(data_root: Path) -> dict:
    counts = {}
    for key, (filename, expected) in COUNT_SCRIPTS.items():
        path = f"analysis_scripts/mechanism_pipeline/{filename}"
        source = _source(data_root / path, path, script=True)
        count, line = _eval_constant(source["document"], path)
        _require(count == expected, f"Unexpected diagnostic count: {path}")
        counts[key] = {"count": count, "verification": "script_constant", "constant": "N_EVAL",
                       "line": line, "source": source}
    return counts


def _validate_record(record: dict, selection: dict) -> None:
    for field in ("seed", "iteration", "checkpoint_path", "metadata_path"):
        value = _field(record, field)
        _require(type(value) is type(selection[field]) and value == selection[field],
                 f"Unexpected checkpoint record: {field}")
    _require(_field(record, "verification_source") ==
             "checkpoint config.seed and iter_num, verified on CPU on 2026-10-06",
             "Unexpected checkpoint verification source")
    _require(_field(record, "verification_date") == "2026-10-06", "Unexpected verification date")


def _validate_heads(document: dict, selection: dict, head_mapping: bool = False) -> None:
    heads = _field(document, "bos_heads")
    _require(isinstance(heads, list) and all(type(h) is int for h in heads)
             and sorted(heads) == selection["recorded_heads"], "Unexpected recorded heads")
    if head_mapping:
        mapping = _field(document, "heads")
        _require(isinstance(mapping, dict), "Invalid head mapping")
        for head in selection["recorded_heads"]:
            _require(isinstance(_field(document, f"heads.h{head}"), dict), f"Invalid head: h{head}")


def _validate_beta(beta: dict, record: dict, selection: dict) -> None:
    path = _field(beta, "adapter_meta.checkpoint_path")
    _require(isinstance(path, str), "Invalid beta checkpoint path")
    expected = Path(record["checkpoint_path"]).parts
    _require(Path(path).parts[-len(expected):] == expected, "Different beta checkpoint path")
    if selection["kind"] == "toy":
        _require(_integer(beta, "adapter_meta.seed") == record["seed"], "Different beta seed")
        _require(_integer(beta, "adapter_meta.iter_num") == record["iteration"], "Different beta iteration")


def _trained_length(beta: dict, record: dict, model_id: str, selection: dict) -> tuple[int, dict]:
    if selection["kind"] == "toy":
        return _integer(beta, "adapter_meta.block_size_train", 1), _ref("beta", "adapter_meta.block_size_train")
    source = _field(record, "trained_context_source")
    _require(_field(source, "path") == selection["context_source"],
             "Different cached metadata source")
    digest = _field(source, "sha256")
    _require(isinstance(digest, str) and len(digest) == 64
             and all(char in "0123456789abcdef" for char in digest), "Invalid cached metadata hash")
    _require(_field(source, "fields") == ["[0].config.block_size", "[0].model_args.block_size"],
             "Different cached training field references")
    _require(_field(source, "selection") == f"seed == {selection['seed']}", "Different cached metadata selection")
    cached = _field(source, "document")
    _require(_integer(cached, "seed") == record["seed"], "Different cached training seed")
    _require(_integer(cached, "config.seed") == record["seed"], "Different cached configuration seed")
    _require(_integer(cached, "iter_num") == record["iteration"], "Different cached iteration")
    _require(_field(cached, "path") == record["checkpoint_path"], "Different cached checkpoint path")
    length = _integer(cached, "config.block_size", 1)
    _require(length == _integer(cached, "model_args.block_size", 1), "Different cached training lengths")
    return length, _ref("checkpoint_record", f"{model_id}.trained_context_source.document.config.block_size")


def _ranges(metrics: list[dict], names: tuple[str, ...]) -> dict:
    return {name: [min(m[name] for m in metrics), max(m[name] for m in metrics)] for name in names}


def _observations(sources: dict, heads: list[int], kind: str) -> dict:
    evidence = sources["evidence"]["document"]
    softmax = sources["softmax"]["document"]
    length = _integer(evidence, "L", 65)
    t_used = _integer(softmax, "T_used", 2)
    _require(t_used <= length, "Softmax window exceeds evidence length")
    positions = _field(evidence, "step1_before_mlp")
    _require(isinstance(positions, dict) and bool(positions), "Missing attention points")
    points = []
    for position in positions:
        _require(isinstance(position, str) and position.isdigit(), "Invalid attention position")
        j = int(position)
        _require(0 <= j < length, "Attention position exceeds evidence length")
        field = f"step1_before_mlp.{position}.layer1_attention_on_BOS_headmean"
        weight = _number(evidence, field)
        _require(0 <= weight <= 1, "Invalid BOS attention probability")
        points.append({"position": j, "weight": weight, "source_ref": _ref("evidence", field)})
    points.sort(key=lambda point: point["position"])
    _require(points[0]["position"] == 0, "Missing attention point at position 0")
    slopes = _field(evidence, "step2_mlp.per_token_slope_at_fixed_j")
    _require(isinstance(slopes, list) and bool(slopes), "Missing fixed-position slopes")
    for slope in slopes:
        _require(_integer(slope, "j") < length, "Slope position exceeds evidence length")
        _number(slope, "slope_correction_on_pre")
        _number(slope, "r2")
    before = _number(evidence, "step2_mlp.late_sd_before")
    after = _number(evidence, "step2_mlp.late_sd_after")
    _require(before >= 0 and after >= 0, "Invalid projection standard deviation")
    noise_status = "absolute_spread_reduced" if after < before else "absolute_spread_not_reduced"
    interpretation = (
        "The stored absolute projection spread decreases after the MLP." if after < before else
        "The stored absolute projection spread does not decrease after the MLP."
    )
    if kind == "lm":
        noise_status = "unmeasured"
        interpretation = (
            "Signal-relative denoising is unmeasured. "
            "Absolute spread alone does not determine signal-relative noise."
        )
    key_metrics, value_metrics = [], []
    for head in heads:
        h = f"h{head}"
        share = f"heads.{h}.b_cen"
        fields = {
            "attention_tv": ("softmax", f"{share}.mean_tv_rows"),
            "B_fitted": ("key_shift", f"B_fit_attention.{head}"),
            "B_measured": ("key_shift", f"/arms/bos_dir_+0.2/causal_B/{h}"),
            "value_r2": ("softmax", f"{share}.value_linear_in_beta_r2"),
            "graded_readout_r2": ("softmax", f"{share}.readout_from_beta_r2"),
        }
        metric = {name: _number(sources[source]["document"], field)
                  for name, (source, field) in fields.items()}
        _require(0 <= metric["attention_tv"] <= 1, "Invalid attention total variation distance")
        for names, output in ((("attention_tv", "B_fitted", "B_measured"), key_metrics),
                              (("value_r2", "graded_readout_r2"), value_metrics)):
            output.append({"head": head, **{name: metric[name] for name in names},
                           "source_refs": {name: _ref(*fields[name]) for name in names}})
    return {
        "1": {"attention_points": points, "attention_summary": "mean_over_first_layer_heads",
              "position_axis": "log2_position_plus_1"},
        "2": {"sd_before": before, "sd_after": after, "slopes": slopes,
              "spread_positions": [64, length - 1],
              "spread_definition": "Mean of per-position projection standard deviations across sequences.",
              "noise_status": noise_status, "interpretation": interpretation,
              "source_refs": {"sd_before": _ref("evidence", "step2_mlp.late_sd_before"),
                              "sd_after": _ref("evidence", "step2_mlp.late_sd_after"),
                              "slopes": _ref("evidence", "step2_mlp.per_token_slope_at_fixed_j")}},
        "3": {"head_metrics": key_metrics,
              "ranges": _ranges(key_metrics, ("attention_tv", "B_fitted", "B_measured")),
              "query_positions": [1, t_used - 1], "key_shift_delta": 0.2,
              "attention_distance_interpretation": "Lower total variation distance means closer agreement."},
        "4": {"head_metrics": value_metrics,
              "ranges": _ranges(value_metrics, ("value_r2", "graded_readout_r2")),
              "query_positions": [1, t_used - 1], "value_key_positions": [0, t_used - 1],
              "value_calibration_key_positions": [1, t_used - 1],
              "prediction": "attention_and_values_from_measured_share"},
    }


def _causal(sources: dict, kind: str) -> dict:
    document = sources["interventions"]["document"]
    lm = kind == "lm"
    causal = {"1": {"status": "not_tested", "explanation":
                        "These artifacts contain no isolated causal test of first-layer BOS attention."}}
    # The MLP edit starts at the first ordinary token, after BOS and any attention sinks.
    first_ordinary = _integer(document, "a", 1)
    for step, (targeted, random) in ARMS_BY_STEP.items():
        arms = {}
        for role, name in (("baseline", "clean"), ("targeted", targeted), ("random", random)):
            field = f"arms.{name}.mae_middle"
            mae = _number(document, field)
            _require(mae >= 0, "Invalid position MAE")
            arms[role] = {"arm": name, "mae_middle": mae,
                          "loss_increase": _number(document, f"arms.{name}.loss_increase") if lm else None,
                          "source_refs": {"mae_middle": _ref("interventions", field)}}
            if lm:
                arms[role]["source_refs"]["loss_increase"] = _ref("interventions", f"arms.{name}.loss_increase")
        causal[step] = {
            "status": "tested", **arms, "positions": [16, 127], "mae_unit": "tokens",
            "edited_positions": "from_first_ordinary" if step == "2" else "all",
            "first_ordinary_position": first_ordinary,
            "first_ordinary_ref": _ref("interventions", "a"),
            "position_output": "frozen_diagnostic_position_probe" if lm else "frozen_trained_position_head",
            "edit_scope": "recomputes_all_downstream_paths" if step == "2" else "all_second_layer_heads",
            "interpretation": "These tests establish use of the projection, not a unique proof of the full mechanism.",
        }
    return causal


def build_snapshot(data_root: Path, records_path: Path) -> dict:
    """Read and validate the selected evidence snapshot."""
    data_root, records_path = Path(data_root), Path(records_path)
    record_source = _source(records_path, "apps/mechanism_evidence/checkpoint_records.json")
    records = record_source["document"]
    _require(isinstance(records, dict) and set(records) == set(MODEL_SELECTION), "Unexpected checkpoint records")
    counts = _counts(data_root)
    models = []
    for model_id, selection in MODEL_SELECTION.items():
        record = records[model_id]
        _validate_record(record, selection)
        folder = f"results/mechanism/story_evidence/{model_id}"
        sources = {key: _source(data_root / folder / filename, f"{folder}/{filename}")
                   for key, filename in DIAGNOSTICS.items()}
        sources["beta"] = _source(data_root / record["metadata_path"], record["metadata_path"])
        sources["checkpoint_record"] = record_source
        for key, source in sources.items():
            if key != "checkpoint_record":
                _require(_field(source["document"], "model") == model_id, f"Different model identifier: {source['path']}")
            if key in ("evidence", "softmax", "key_shift", "beta"):
                _validate_heads(source["document"], selection, head_mapping=key in ("evidence", "softmax"))
        beta = sources["beta"]["document"]
        _validate_beta(beta, record, selection)
        _require(_integer(sources["evidence"]["document"], "n_eval_seqs", 1) == counts["story"]["count"],
                 "Story JSON count differs from the script constant")
        trained_length, trained_ref = _trained_length(beta, record, model_id, selection)
        _require(trained_length == selection["trained_context_length"], "Different selected training context length")
        models.append({
            "id": model_id, "name": selection["name"], "seed": record["seed"], "iteration": record["iteration"],
            "checkpoint_path": record["checkpoint_path"], "heads": list(selection["heads"]),
            "recorded_heads": list(selection["recorded_heads"]),
            "checkpoint_verification": {"date": record["verification_date"], "source": record["verification_source"]},
            "trained_context_length": trained_length,
            "evidence_context_length": _integer(sources["evidence"]["document"], "L", 65),
            "softmax_T_used": _integer(sources["softmax"]["document"], "T_used", 2),
            "source_refs": {"trained_context_length": trained_ref,
                            "evidence_context_length": _ref("evidence", "L"),
                            "softmax_T_used": _ref("softmax", "T_used")},
            "observations": _observations(sources, selection["heads"], selection["kind"]),
            "causal": _causal(sources, selection["kind"]), "sources": sources,
        })
    return {
        "schema_version": 1,
        "protocol": {
            "counts": counts, "middle_positions": [16, 127], "share_field": "b_cen",
            "selected_examples": "These selected examples do not establish that every seed follows the mechanism.",
            "graded_share": "A graded share is valid. The two-level pattern is a special case.",
            "spread_note": "The spread summary uses positions 64 onward, not the manuscript late bin.",
            "count_note": "Diagnostic counts come from script constants, not beta extraction split sizes.",
            "flat_share_note": "The LM-BOS flat-share edit is a signal ablation, not a preserving control.",
            "metric_definitions": {
                "attention_tv": "Mean total variation distance over query positions 1 through T_used minus 1.",
                "value_r2": "1 minus residual variance divided by measured value variance, over keys 0 through T_used minus 1.",
                "graded_readout_r2": "1 minus residual variance divided by measured variance, not mean squared error.",
                "graded_prediction": "Predict both attention weights and value projections from the measured share.",
            },
        },
        "models": models,
    }


def render_html(snapshot: dict, assets_dir: Path) -> str:
    """Embed data and local assets in a portable HTML document."""
    assets_dir = Path(assets_dir)
    payload = json.dumps(snapshot, allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    assets = {
        "CSS": (assets_dir / "styles.css").read_text(encoding="utf-8"),
        "DATA": payload,
        "JS": (assets_dir / "app.js").read_text(encoding="utf-8"),
    }
    template = (assets_dir / "template.html").read_text(encoding="utf-8")
    # Substitute only template tokens, never tokens inside inserted source documents.
    return re.sub(r"\{\{(CSS|DATA|JS)\}\}", lambda match: assets[match[1]], template)


def build_app(data_root: Path, output_path: Path, assets_dir: Path) -> None:
    """Validate all data before replacing the output file."""
    assets_dir, output_path = Path(assets_dir), Path(output_path)
    snapshot = build_snapshot(Path(data_root), assets_dir / "checkpoint_records.json")
    html = render_html(snapshot, assets_dir)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent,
                                         prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as output:
            temporary_path = Path(output.name)
            output.write(html)
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    """Build the app from an explicit cached data root."""
    assets_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Read-only cached evidence root")
    parser.add_argument("--output", type=Path, default=assets_dir / "index.html", help="Portable HTML destination")
    args = parser.parse_args()
    build_app(args.data_root, args.output, assets_dir)
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
