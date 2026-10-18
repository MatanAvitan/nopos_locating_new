# Mechanism evidence app

A single offline HTML page that shows the evidence for each of the four mechanism steps.
It covers three selected checkpoints: 1H (seed 42, iteration 19500), 12H (seed 1, iteration 17500), and LM-BOS (seed 9, iteration 20000).
For each step, the page separates observed evidence from causal evidence and keeps the source records in expandable panels.

## Build

The exporter reads cached JSON only. It loads no model and needs no GPU.

```bash
python apps/mechanism_evidence/export_data.py \
  --data-root /home/nlp/matan_avitan/git/nopos_locating_new \
  --output apps/mechanism_evidence/index.html
```

Open `index.html` directly in a browser. It needs no server and makes no network request.

## Test

```bash
MECHANISM_DATA_ROOT=/home/nlp/matan_avitan/git/nopos_locating_new \
python -m unittest discover -s apps/mechanism_evidence/tests -v
```

The browser tests need Playwright with Chromium. They open the page through a file URL with network access blocked.

## Files

- `export_data.py` validates the cached evidence and embeds it in the page.
- `checkpoint_records.json` holds the seed and iteration verified from each checkpoint.
- `template.html`, `styles.css`, and `app.js` are the page.
- `tests/` holds the exporter and browser tests.
