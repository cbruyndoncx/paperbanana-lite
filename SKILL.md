---
name: paperbanana-lite
description: Generate publication-quality academic diagrams and statistical plots from text using a multi-agent Gemini pipeline.
entry: paperbanana_lite
category: ai-image
---

# PaperBanana Lite Skill

Generate publication-quality methodology diagrams and statistical plots from text descriptions using a multi-agent pipeline powered by Google Gemini. Single-file version of the full PaperBanana framework.

## Usage

```bash
python paperbanana_lite.py generate --input methodology.txt --caption "Overview of our architecture"
python paperbanana_lite.py plot --data results.json --intent "Bar chart comparing model accuracy"
```

## Examples

### Generate a methodology diagram

```bash
python paperbanana_lite.py generate \
  --input methodology.txt \
  --caption "Overview of our encoder-decoder architecture" \
  --reference-dir data/reference_sets \
  --iterations 3
```

### Generate a statistical plot

```bash
python paperbanana_lite.py plot \
  --data data.json \
  --intent "Bar chart comparing model accuracy across datasets" \
  --reference-dir data/reference_sets \
  --iterations 3
```

## Options

### `generate` subcommand (methodology diagrams)

| Option | Purpose |
|--------|---------|
| `--input` | Path to text file with methodology section (required) |
| `--caption` | Figure caption / communicative intent (required) |
| `--reference-dir` | Path to reference set directory (default: `data/reference_sets`) |
| `--iterations` | Number of refinement iterations (default: 3) |
| `--output-dir` | Base output directory (default: `outputs`) |

### `plot` subcommand (statistical plots)

| Option | Purpose |
|--------|---------|
| `--data` | Path to JSON file with raw data (required) |
| `--intent` | Visual intent / figure caption (required) |
| `--reference-dir` | Path to reference set directory (default: `data/reference_sets`) |
| `--iterations` | Number of refinement iterations (default: 3) |
| `--output-dir` | Base output directory (default: `outputs`) |

## Environment Variables

```bash
export GEMINI_API_KEY="your-key"
```

Also accepts `GOOGLE_API_KEY` as a fallback.

## Dependencies

Only 3 external packages:

```
pip install google-genai pillow tenacity
```

## How It Works

Two-phase multi-agent pipeline:

**Phase 1 — Linear Planning:**
1. **Retriever** — selects relevant reference examples from curated set via VLM
2. **Planner** — generates detailed textual description via in-context learning
3. **Stylist** — refines description with NeurIPS-style aesthetic guidelines

**Phase 2 — Iterative Refinement (up to N iterations):**
4. **Visualizer** — renders description into image (Gemini image gen for diagrams, matplotlib code for plots)
5. **Critic** — evaluates image quality, provides revision feedback; loops back to step 4

## Output

Results are saved to `outputs/run_<timestamp>/` containing:
- `final_output.png` — the final generated image
- `planning.json` — retrieved examples and initial description
- `iter_N.png` — image from each iteration
- `iter_N_details.json` — description and critic feedback per iteration

## Reference Sets

The `--reference-dir` should point to a directory containing an `index.json` file with curated reference examples. The default reference set ships with the repo at `data/reference_sets/` (13 methodology diagrams).
