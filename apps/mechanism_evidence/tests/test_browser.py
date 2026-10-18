"""Check the portable interface through an offline file URL."""

import copy
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

from playwright.sync_api import sync_playwright

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
import export_data

DATA_ROOT = Path(os.environ["MECHANISM_DATA_ROOT"])


class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "index.html"
        export_data.build_app(DATA_ROOT, cls.output, APP_DIR)
        cls.snapshot = export_data.build_snapshot(DATA_ROOT, APP_DIR / "checkpoint_records.json")
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.temp.cleanup()

    def setUp(self):
        self.context = self.browser.new_context(offline=True, viewport={"width": 1440, "height": 1000})
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.errors = []
        self.requests = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on("console", lambda message: self.errors.append(message.text) if message.type == "error" else None)
        self.page.on("request", lambda request: self.requests.append(request.url))
        self.page.goto(self.output.as_uri())

    def assert_offline_clean(self):
        self.assertEqual(self.errors, [])
        self.assertEqual(self.requests, [self.output.as_uri()])

    def test_steps_filters_keyboard_and_sources(self):
        page = self.page
        self.assertEqual(page.locator(".model-card:visible").count(), len(self.snapshot["models"]))
        page.get_by_role("tab", name="3 Keys", exact=True).click()
        self.assertTrue(page.locator("#step-title").inner_text().startswith("3"))
        lm = next(m for m in self.snapshot["models"] if m["id"] == "lmbos_s9")
        page.get_by_role("button", name=f'{lm["name"]}, seed {lm["seed"]}', exact=True).click()
        self.assertEqual(page.locator(".model-card:visible").count(), 1)
        for number, title in (("1", "Attention"), ("2", "MLP"), ("3", "Keys"), ("4", "Values")):
            tab = page.get_by_role("tab", name=f"{number} {title}", exact=True)
            tab.click()
            self.assertEqual(tab.get_attribute("aria-selected"), "true")
            self.assertEqual(tab.get_attribute("aria-controls"), "evidence-panel")
            self.assertEqual(tab.get_attribute("tabindex"), "0")
            for model in self.snapshot["models"]:
                button = page.get_by_role("button", name=f'{model["name"]}, seed {model["seed"]}', exact=True)
                button.click()
                self.assertEqual(button.get_attribute("aria-pressed"), "true")
                card = page.locator(".model-card")
                self.assertEqual(card.count(), 1)
                self.assertIn(model["name"], card.inner_text())
                self.assertIn("Observed evidence", card.inner_text())
                self.assertIn("Causal evidence", card.inner_text())
                summary = card.get_by_text("Measurements and source records", exact=True)
                summary.focus()
                page.keyboard.press("Enter")
                card.get_by_text("softmax source", exact=True).click()
                card.get_by_text("Observed measurements and exact JSON field references", exact=True).click()
                card.get_by_text("Causal arms and exact JSON field references", exact=True).click()
                detail_text = card.locator(".source-details").inner_text()
                self.assertIn(model["checkpoint_path"], detail_text)
                self.assertIn(str(model["iteration"]), detail_text)
                self.assertIn(model["sources"]["softmax"]["sha256"], detail_text)
                if number == "4":
                    self.assertIn(f'/heads/h{model["heads"][0]}/b_cen/readout_from_beta_r2', detail_text)
                    self.assertIn(str(model["observations"]["4"]["head_metrics"][0]["graded_readout_r2"]), detail_text)
            page.get_by_role("button", name="All models", exact=True).click()
            self.assertEqual(page.locator(".model-card:visible").count(), len(self.snapshot["models"]))
        page.get_by_role("tab", name="4 Values", exact=True).focus()
        page.keyboard.press("ArrowRight")
        self.assertEqual(page.get_by_role("tab", name="1 Attention").get_attribute("aria-selected"), "true")
        page.keyboard.press("ArrowLeft")
        self.assertEqual(page.get_by_role("tab", name="4 Values").get_attribute("aria-selected"), "true")
        page.keyboard.press("Home")
        self.assertTrue(page.locator("#step-title").inner_text().startswith("1"))
        page.keyboard.press("End")
        self.assertTrue(page.locator("#step-title").inner_text().startswith("4"))
        model_button = page.get_by_role("button", name="1H, seed 42", exact=True)
        model_button.focus()
        page.keyboard.press("Space")
        self.assertEqual(model_button.get_attribute("aria-pressed"), "true")
        self.assertEqual(page.locator(".model-card:visible").count(), 1)
        self.assert_offline_clean()

    def test_scientific_labels_numeric_charts_and_loss_rule(self):
        page = self.page
        self.assertEqual(page.get_by_text("No isolated causal test", exact=True).count(), len(self.snapshot["models"]))
        self.assertEqual(page.get_by_text("Starts at BOS, position 0, where the weight is 1.", exact=True).count(), len(self.snapshot["models"]))
        self.assertIn("not a zero effect", page.locator("#step-notes").inner_text())
        # A positive probability below 0.001 must not display as zero.
        self.assertIn("<0.001", page.locator(".model-card").first.locator(".metrics").inner_text())
        model = self.snapshot["models"][0]
        points = model["observations"]["1"]["attention_points"]
        self.assertEqual(points[0]["position"], 0)
        path = page.locator('.model-card').first.locator('svg .attention-line').get_attribute("d")
        expected = " ".join(f'{"M" if i == 0 else "L"}{42 + 260 * math.log2(p["position"] + 1) / math.log2(max(q["position"] for q in points) + 1)},{16 + 140 * (1 - p["weight"])}' for i, p in enumerate(points))
        # JavaScript and Python can differ in the last floating-point digit.
        self.assertEqual(len(re.findall(r"[ML]", path)), len(points))
        for actual, reference in zip(re.findall(r"[\d.]+", path), re.findall(r"[\d.]+", expected)):
            self.assertAlmostEqual(float(actual), float(reference), places=9)
        for step, title in (("2", "MLP"), ("3", "Keys"), ("4", "Values")):
            page.get_by_role("tab", name=f"{step} {title}").click()
            for index, model in enumerate(self.snapshot["models"]):
                card = page.locator(".model-card").nth(index)
                for role in ("baseline", "targeted", "random"):
                    mae = model["causal"][step][role]["mae_middle"]
                    self.assertIn(f"{mae:.2f} tokens", card.inner_text())
                self.assertEqual(card.locator(".causal-chart .bar").count(), 3)
                lm = model["id"].startswith("lmbos")
                self.assertIn("frozen diagnostic position probe" if lm else "frozen trained position head", card.inner_text())
                if not lm:
                    self.assertNotIn("Next-token loss", card.inner_text())
                where = ("at every position" if step != "2"
                         else f'from position {model["causal"]["2"]["first_ordinary_position"]} on')
                self.assertIn(where, card.locator(".edit-line").inner_text())
            if step == "2":
                for k in (2, 3):
                    self.assertIn("Signal-relative denoising is unmeasured", page.locator(".model-card").nth(k).inner_text())
                self.assertIn("Correction slopes", page.locator(".model-card").first.inner_text())
            if step == "3":
                self.assertIn("ranges across the six selected heads", page.locator("#step-notes").inner_text())
                self.assertIn("log scale", page.locator(".causal-chart").first.text_content())
            self.assertIn("Random control", page.locator("#step-notes").inner_text())
            if step == "4":
                self.assertIn("<0.001", page.locator(".model-card").nth(2).inner_text())
                self.assertIn("both attention", page.locator("#prediction").inner_text())
        self.assert_offline_clean()

    def test_numeric_bar_and_head_coordinates(self):
        page = self.page
        for step, title in (("2", "MLP"), ("3", "Keys"), ("4", "Values")):
            page.get_by_role("tab", name=f"{step} {title}").click()
            for index, model in enumerate(self.snapshot["models"]):
                card = page.locator(".model-card").nth(index)
                causal = model["causal"][step]
                values = [causal[role]["mae_middle"] for role in ("baseline", "targeted", "random")]
                log_scale = min(values) > 0 and max(values) / min(values) > 20
                if log_scale:
                    low = 10 ** math.floor(math.log10(min(values))) / 10
                    high = 10 ** math.ceil(math.log10(max(values)))
                    widths = [230 * (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low)) for value in values]
                else:
                    widths = [230 * value / (max(values) * 1.12) for value in values]
                for bar, width in zip(card.locator(".causal-chart .bar").all(), widths):
                    self.assertAlmostEqual(float(bar.get_attribute("width")), width, places=9)
                observation = model["observations"][step]
                if step == "2":
                    values = [observation["sd_before"], observation["sd_after"]]
                    for bar, value in zip(card.locator(".spread-chart .bar").all(), values):
                        self.assertAlmostEqual(float(bar.get_attribute("width")), 230 * value / (max(values) * 1.12), places=9)
                else:
                    fields = ("B_fitted", "B_measured") if step == "3" else ("value_r2", "graded_readout_r2")
                    values = [head[field] for head in observation["head_metrics"] for field in fields]
                    low = min(0, *values)
                    high = max(1, *values) * (1.1 if step == "3" else 1)
                    circles = card.locator(".head-chart circle").all()[1:]
                    squares = card.locator(".head-chart rect").all()[1:]
                    self.assertEqual(len(circles), len(model["heads"]))
                    for head, circle, square in zip(observation["head_metrics"], circles, squares):
                        self.assertAlmostEqual(float(circle.get_attribute("cx")), 48 + 260 * (head[fields[0]] - low) / (high - low), places=9)
                        self.assertAlmostEqual(float(square.get_attribute("x")) + 4, 48 + 260 * (head[fields[1]] - low) / (high - low), places=9)
        self.assert_offline_clean()

    def test_narrow_layout_and_screenshots(self):
        page = self.page
        screenshot_dir = Path(os.environ.get("MECHANISM_SCREENSHOT_DIR", self.temp.name))
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        for number, title in (("1", "Attention"), ("2", "MLP"), ("3", "Keys"), ("4", "Values")):
            page.get_by_role("tab", name=f"{number} {title}").click()
            page.screenshot(path=str(screenshot_dir / f"desktop-step-{number}.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        for number, title in (("1", "Attention"), ("2", "MLP"), ("3", "Keys"), ("4", "Values")):
            page.get_by_role("tab", name=f"{number} {title}").click()
            self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 390)
            page.screenshot(path=str(screenshot_dir / f"mobile-step-{number}.png"), full_page=True)
            for svg in page.locator("svg").all():
                box = svg.bounding_box()
                self.assertGreater(box["width"], 200)
                self.assertLessEqual(box["x"] + box["width"], 390)
        for card in page.locator(".model-card").all():
            card.get_by_text("Measurements and source records", exact=True).click()
            card.get_by_text("softmax source", exact=True).click()
        page.get_by_text("Protocol, counts, and metric definitions", exact=True).click()
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 390)
        self.assert_offline_clean()

    def test_artifact_text_cannot_execute(self):
        snapshot = copy.deepcopy(self.snapshot)
        attack = '</script><script>window.injected=true</script><img src="https://example.invalid/x">'
        snapshot["models"][0]["name"] = attack
        snapshot["models"][0]["sources"]["evidence"]["document"]["test_string"] = attack
        path = Path(self.temp.name) / "hostile.html"
        path.write_text(export_data.render_html(snapshot, APP_DIR))
        self.page.goto(path.as_uri())
        self.assertEqual(self.page.locator("script").count(), 2)
        self.assertEqual(self.page.locator("img").count(), 0)
        self.assertIsNone(self.page.evaluate("window.injected"))
        self.assertIn(attack, self.page.locator(".model-card").first.inner_text())
        self.assertEqual(self.errors, [])
        self.assertEqual(self.requests, [self.output.as_uri(), path.as_uri()])


if __name__ == "__main__":
    unittest.main()
