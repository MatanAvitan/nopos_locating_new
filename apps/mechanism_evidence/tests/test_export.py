"""Test the exporter with the selected cached evidence."""

import ast
import copy
from html.parser import HTMLParser
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import sys
import unittest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
import export_data
from export_data import build_snapshot

DATA_ROOT = Path(os.environ.get("MECHANISM_DATA_ROOT", APP_DIR.parents[1]))
RECORDS_PATH = APP_DIR / "checkpoint_records.json"


class CanonicalTests(unittest.TestCase):
    def test_canonical_values(self):
        snapshot = build_snapshot(DATA_ROOT, RECORDS_PATH)
        models = {model["id"]: model for model in snapshot["models"]}
        self.assertEqual([(m["seed"], m["iteration"]) for m in snapshot["models"]],
                         [(42, 19500), (1, 17500), (9, 20000), (2, 20000)])
        self.assertEqual(models["1h"]["causal"]["1"]["status"], "not_tested")
        self.assertEqual(models["lmbos_s9"]["observations"]["2"]["noise_status"], "unmeasured")
        self.assertEqual(models["lmbos_L1024_s2"]["observations"]["2"]["noise_status"], "unmeasured")
        self.assertEqual(models["lmbos_L1024_s2"]["heads"], [7])
        self.assertAlmostEqual(models["lmbos_s9"]["observations"]["4"]["head_metrics"][0][
            "graded_readout_r2"], 0.9472177947096545, delta=1e-12)
        self.assertAlmostEqual(models["1h"]["causal"]["3"]["targeted"]["mae_middle"],
                               1009.4657546775411, delta=1e-9)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = build_snapshot(DATA_ROOT, RECORDS_PATH)

    def test_counts_and_protocol(self):
        protocol = self.snapshot["protocol"]
        self.assertEqual({k: v["count"] for k, v in protocol["counts"].items()},
                         {"story": 400, "softmax": 400, "key_shift": 200, "interventions": 1600})
        for item in protocol["counts"].values():
            self.assertEqual(item["verification"], "script_constant")
            self.assertEqual(item["constant"], "N_EVAL")
            ast.parse(item["source"]["document"])
        self.assertIn("residual variance", protocol["metric_definitions"]["graded_readout_r2"])
        self.assertIn("measured variance", protocol["metric_definitions"]["graded_readout_r2"])
        self.assertEqual(protocol["middle_positions"], [16, 127])

    def test_heads_ranges_and_sources(self):
        for model, heads in zip(self.snapshot["models"], ([0], [2, 3, 5, 9, 10, 11], [0], [7])):
            self.assertEqual(model["heads"], heads)
            self.assertEqual(model["checkpoint_verification"]["date"], "2026-10-06")
            for step in ("3", "4"):
                obs = model["observations"][step]
                self.assertEqual([m["head"] for m in obs["head_metrics"]], heads)
                for metric, limits in obs["ranges"].items():
                    values = [m[metric] for m in obs["head_metrics"]]
                    self.assertEqual(limits, [min(values), max(values)])
            for key in ("evidence", "softmax", "key_shift", "interventions", "beta"):
                src = model["sources"][key]
                raw = (DATA_ROOT / src["path"]).read_bytes()
                self.assertEqual(src["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(src["document"], json.loads(raw))
            record_source = model["sources"]["checkpoint_record"]
            self.assertEqual(record_source["document"], json.loads(RECORDS_PATH.read_text()))
            self.assertEqual(record_source["sha256"], hashlib.sha256(RECORDS_PATH.read_bytes()).hexdigest())
            for name, ref in model["source_refs"].items():
                value = model["sources"][ref["source"]]["document"]
                for part in ref["field"][1:].split("/"):
                    value = value[part]
                self.assertEqual(value, model[name])
            for step in ("3", "4"):
                for metric in model["observations"][step]["head_metrics"]:
                    for name, ref in metric["source_refs"].items():
                        value = model["sources"][ref["source"]]["document"]
                        for part in ref["field"][1:].split("/"):
                            value = value[part]
                        self.assertEqual(value, metric[name])

    def test_observation_scope_and_causal_mapping(self):
        for model in self.snapshot["models"]:
            obs = model["observations"]
            self.assertEqual(obs["1"]["attention_summary"], "mean_over_first_layer_heads")
            self.assertEqual(obs["1"]["position_axis"], "log2_position_plus_1")
            self.assertEqual(obs["1"]["attention_points"][0]["position"], 0)
            self.assertEqual(obs["2"]["spread_positions"], [64, model["evidence_context_length"] - 1])
            self.assertEqual(obs["3"]["query_positions"], [1, model["softmax_T_used"] - 1])
            self.assertEqual(obs["4"]["value_key_positions"], [0, model["softmax_T_used"] - 1])
            self.assertEqual(obs["4"]["value_calibration_key_positions"], [1, model["softmax_T_used"] - 1])
            self.assertEqual(obs["4"]["prediction"], "attention_and_values_from_measured_share")
            self.assertEqual(model["trained_context_length"], 128 if model["id"] == "lmbos_s9" else 1024)
            lm = model["id"].startswith("lmbos")
            self.assertEqual(model["softmax_T_used"], 128 if model["id"] == "lmbos_s9" else 256)
            self.assertNotIn("baseline", model["causal"]["1"])
            for step, arm in (("2", "mlp_mean"), ("3", "keys_noshare"), ("4", "values_noshare")):
                causal = model["causal"][step]
                # Position 1 is an attention sink in 1H and 12H, so their MLP edit starts at 2.
                self.assertEqual(causal["first_ordinary_position"], 1 if lm else 2)
                self.assertEqual(causal["first_ordinary_ref"], {"source": "interventions", "field": "/a"})
                self.assertEqual(causal["edited_positions"], "from_first_ordinary" if step == "2" else "all")
                for role, name in (("baseline", "clean"), ("targeted", arm), ("random", arm + "_rand")):
                    result = causal[role]
                    self.assertEqual(result["arm"], name)
                    doc = model["sources"]["interventions"]["document"]["arms"][name]
                    self.assertEqual(result["mae_middle"], doc["mae_middle"])
                    self.assertEqual(result["loss_increase"], doc.get("loss_increase"))
        json.dumps(self.snapshot, allow_nan=False)


class InvalidInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = json.loads(RECORDS_PATH.read_text())
        self.records_path = self.root / "checkpoint_records.json"
        self.records_path.write_text(json.dumps(self.records))
        for model, record in self.records.items():
            folder = Path("results/mechanism/story_evidence") / model
            for name in ("evidence.json", "softmax_beta_check.json", "causal_story.json", "denoise_interventions_v2.json"):
                self.copy(folder / name)
            self.copy(Path(record["metadata_path"]))
        for name in ("story_evidence", "softmax_beta_check", "causal_story", "denoise_interventions_v2"):
            self.copy(Path("analysis_scripts/mechanism_pipeline") / (name + ".py"))

    def copy(self, path):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA_ROOT / path, target)

    def mutate(self, path, change):
        target = self.root / path
        doc = json.loads(target.read_text())
        change(doc)
        target.write_text(json.dumps(doc))

    def diagnostic(self, name, model="1h"):
        return Path("results/mechanism/story_evidence") / model / name

    def assert_invalid(self):
        with self.assertRaises(ValueError):
            build_snapshot(self.root, self.records_path)

    def test_copied_fixtures_and_absolute_path_prefix(self):
        self.mutate(self.records["1h"]["metadata_path"], lambda d: d["adapter_meta"].update(
            checkpoint_path="/copied/research/" + self.records["1h"]["checkpoint_path"]))
        self.assertEqual(len(build_snapshot(self.root, self.records_path)["models"]), len(self.records))

    def test_wrong_model_identifier(self):
        for name in ("evidence.json", "softmax_beta_check.json", "causal_story.json", "denoise_interventions_v2.json"):
            with self.subTest(name=name):
                path = self.diagnostic(name)
                original = (self.root / path).read_text()
                self.mutate(path, lambda d: d.update(model="wrong"))
                self.assert_invalid()
                (self.root / path).write_text(original)

    def test_missing_selected_head(self):
        for name in ("evidence.json", "softmax_beta_check.json"):
            with self.subTest(name=name):
                path = self.diagnostic(name, "seed_12h_s1")
                original = (self.root / path).read_text()
                self.mutate(path, lambda d: d["heads"].pop("h11"))
                self.assert_invalid()
                (self.root / path).write_text(original)

    def test_missing_recorded_head(self):
        self.mutate(self.diagnostic("evidence.json", "lmbos_s9"), lambda d: d["bos_heads"].remove(3))
        self.assert_invalid()

    def test_different_checkpoint_path(self):
        for model, record in self.records.items():
            with self.subTest(model=model):
                path = record["metadata_path"]
                original = (self.root / path).read_text()
                self.mutate(path, lambda d: d["adapter_meta"].update(checkpoint_path="/other/ckpt.pt"))
                self.assert_invalid()
                (self.root / path).write_text(original)

    def test_incomplete_checkpoint_suffix(self):
        self.mutate(self.records["1h"]["metadata_path"], lambda d: d["adapter_meta"].update(
            checkpoint_path="/other/out-mechanism-ATTN2FULL-1024-lr3e4/R0/ppe56vt4/best_ckpt.pt"))
        self.assert_invalid()

    def test_missing_key(self):
        self.mutate(self.diagnostic("softmax_beta_check.json"), lambda d: d["heads"]["h0"]["b_cen"].pop(
            "readout_from_beta_r2"))
        self.assert_invalid()

    def test_nonfinite_measurement(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.mutate(self.diagnostic("softmax_beta_check.json"), lambda d: d["heads"]["h0"]["b_cen"].update(
                    readout_from_beta_r2=value))
                self.assert_invalid()

    def test_negative_finite_r_squared_is_valid(self):
        self.mutate(self.diagnostic("softmax_beta_check.json"), lambda d: d["heads"]["h0"]["b_cen"].update(
            readout_from_beta_r2=-0.75, value_linear_in_beta_r2=-0.25))
        model = build_snapshot(self.root, self.records_path)["models"][0]
        self.assertEqual(model["observations"]["4"]["head_metrics"][0]["graded_readout_r2"], -0.75)

    def test_increased_finite_spread_does_not_claim_reduction(self):
        self.mutate(self.diagnostic("evidence.json"), lambda d: d["step2_mlp"].update(late_sd_after=10.0))
        obs = build_snapshot(self.root, self.records_path)["models"][0]["observations"]["2"]
        self.assertEqual(obs["noise_status"], "absolute_spread_not_reduced")
        self.assertNotIn("decreases", obs["interpretation"])

    def test_wrong_beta_seed_or_iteration(self):
        for key in ("seed", "iter_num"):
            with self.subTest(key=key):
                path = self.records["1h"]["metadata_path"]
                original = (self.root / path).read_text()
                self.mutate(path, lambda d: d["adapter_meta"].update({key: 999}))
                self.assert_invalid()
                (self.root / path).write_text(original)

    def test_wrong_verified_record(self):
        self.records["lmbos_s9"]["seed"] = 10
        self.records_path.write_text(json.dumps(self.records))
        self.assert_invalid()

    def test_wrong_story_count(self):
        self.mutate(self.diagnostic("evidence.json"), lambda d: d.update(n_eval_seqs=640))
        self.assert_invalid()

    def test_wrong_script_count(self):
        path = self.root / "analysis_scripts/mechanism_pipeline/story_evidence.py"
        path.write_text("N_EVAL, N_CAL = 640, 160\n")
        self.assert_invalid()

    def test_script_constants_without_execution(self):
        path = self.root / "analysis_scripts/mechanism_pipeline/story_evidence.py"
        path.write_text("raise RuntimeError('Do not execute this script')\nN_EVAL = 400\n")
        self.assertEqual(len(build_snapshot(self.root, self.records_path)["models"]), len(self.records))

    def test_missing_cached_training_provenance(self):
        self.records["lmbos_s9"]["trained_context_source"].pop("path")
        self.records_path.write_text(json.dumps(self.records))
        self.assert_invalid()

    def test_different_cached_training_length(self):
        doc = self.records["lmbos_s9"]["trained_context_source"]["document"]
        doc["config"]["block_size"] = 999
        doc["model_args"]["block_size"] = 999
        self.records_path.write_text(json.dumps(self.records))
        self.assert_invalid()

    def test_missing_loss_increase(self):
        self.mutate(self.diagnostic("denoise_interventions_v2.json", "lmbos_s9"),
                    lambda d: d["arms"]["mlp_mean"].pop("loss_increase"))
        self.assert_invalid()

    def test_nonfinite_required_metrics(self):
        cases = [
            ("evidence.json", "1h", lambda d: d["step2_mlp"].update(late_sd_after=float("nan"))),
            ("evidence.json", "1h", lambda d: d["step2_mlp"]["per_token_slope_at_fixed_j"][0].update(r2=float("inf"))),
            ("causal_story.json", "1h", lambda d: d["arms"]["bos_dir_+0.2"]["causal_B"].update(h0=float("nan"))),
            ("denoise_interventions_v2.json", "1h", lambda d: d["arms"]["clean"].update(mae_middle=float("inf"))),
            ("denoise_interventions_v2.json", "lmbos_s9", lambda d: d["arms"]["keys_noshare"].update(loss_increase=float("nan"))),
        ]
        for name, model, change in cases:
            with self.subTest(name=name, model=model):
                path = self.diagnostic(name, model)
                original = (self.root / path).read_text()
                self.mutate(path, change)
                self.assert_invalid()
                (self.root / path).write_text(original)

    def test_missing_or_duplicate_script_constant(self):
        path = self.root / "analysis_scripts/mechanism_pipeline/story_evidence.py"
        for text in ("N_CAL = 400\n", "N_EVAL = 400\nN_EVAL = 400\n", "N_EVAL = 400\nN_EVAL += 1\n"):
            with self.subTest(text=text):
                path.write_text(text)
                self.assert_invalid()

    def test_invalid_json(self):
        (self.root / self.diagnostic("evidence.json")).write_text("{broken")
        self.assert_invalid()

    def test_missing_source_file(self):
        (self.root / self.diagnostic("causal_story.json")).unlink()
        self.assert_invalid()


class DocumentParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.payload = ""
        self.in_payload = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            attributes = dict(attrs)
            self.scripts.append(attributes)
            self.in_payload = attributes.get("id") == "evidence-data"

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_payload = False

    def handle_data(self, data):
        if self.in_payload:
            self.payload += data


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = export_data.build_snapshot(DATA_ROOT, RECORDS_PATH)

    def test_safe_single_pass_embedding(self):
        snapshot = copy.deepcopy(self.snapshot)
        attack = '</script><script>alert(1)</script>&<> {{CSS}} {{DATA}} {{JS}}'
        snapshot["models"][0]["sources"]["evidence"]["document"]["test_string"] = attack
        html = export_data.render_html(snapshot, APP_DIR)
        parsed = DocumentParser()
        parsed.feed(html)
        self.assertEqual(len(parsed.scripts), 2)
        self.assertNotIn("</script><script>alert(1)", html)
        self.assertIn("\\u003c/script\\u003e", parsed.payload)
        self.assertEqual(json.loads(parsed.payload), snapshot)
        self.assertNotIn('<script src=', html)
        self.assertNotIn('<link ', html)

    def test_nonfinite_payload_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["models"][0]["observations"]["2"]["sd_before"] = float("nan")
        with self.assertRaises(ValueError):
            export_data.render_html(snapshot, APP_DIR)

    def test_atomic_build_and_failed_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "index.html"
            output.write_text("existing document")
            with self.assertRaises(ValueError):
                export_data.build_app(Path(temp), output, APP_DIR)
            self.assertEqual(output.read_text(), "existing document")
            records = Path(temp) / "checkpoint_records.json"
            records.write_bytes(RECORDS_PATH.read_bytes())
            with self.assertRaises(FileNotFoundError):
                export_data.build_app(DATA_ROOT, output, Path(temp))
            records.unlink()
            self.assertEqual(output.read_text(), "existing document")
            self.assertEqual(list(Path(temp).iterdir()), [output])
            export_data.build_app(DATA_ROOT, output, APP_DIR)
            self.assertIn('id="evidence-data"', output.read_text())
            self.assertEqual(list(Path(temp).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
