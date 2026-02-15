---
name: paperbanana-lite
description: Generate publication-quality academic methodology diagrams and statistical plots from text descriptions using a multi-agent Gemini pipeline. Use when the user wants to create scientific figures, methodology diagrams, architecture diagrams, or statistical plots for academic papers. Triggers on requests like "generate a diagram", "create a methodology figure", "make a plot for my paper", or "illustrate this architecture".
---

# PaperBanana Lite

Single-file multi-agent pipeline for academic illustration generation using Google Gemini. Distills the full PaperBanana framework into one script.

## Quick Start

```bash
# Download reference dataset (one-time setup, or done automatically on first run)
uv run scripts/paperbanana_lite.py setup

# Diagram generation
uv run scripts/paperbanana_lite.py generate \
  --input methodology.txt \
  --caption "Overview of our encoder-decoder architecture"

# Plot generation
uv run scripts/paperbanana_lite.py plot \
  --data results.json \
  --intent "Bar chart comparing model accuracy"
```

## Environment

Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY` as fallback):

```bash
export GEMINI_API_KEY="your-key"
```

Install dependencies:

```bash
pip install google-genai pillow tenacity
```

## Pipeline Overview

**Phase 1 — Linear Planning:**
1. **Retriever** — selects relevant reference examples from curated set via VLM
2. **Planner** — generates detailed textual description via in-context learning
3. **Stylist** — refines description with NeurIPS-style aesthetic guidelines

**Phase 2 — Iterative Refinement (up to N iterations):**
4. **Visualizer** — renders image (Gemini image gen for diagrams, matplotlib code for plots)
5. **Critic** — evaluates quality, provides revision feedback; loops back to step 4

## CLI Options

### `setup` (download references)

Downloads the curated reference dataset (~937 KB) from GitHub to a local cache. This is optional — references are fetched automatically on first `generate` or `plot` run if not already present.

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--target-dir` | No | `~/.paperbanana/reference_sets` | Where to store references |

Reference lookup order during generation:
1. `--reference-dir` if explicitly provided
2. `data/reference_sets` (local repo directory)
3. `~/.paperbanana/reference_sets` (shared cache)
4. Auto-downloads from GitHub if none found

### `generate` (methodology diagrams)

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--input` | Yes | — | Text file with methodology section |
| `--caption` | Yes | — | Figure caption |
| `--reference-dir` | No | `data/reference_sets` | Reference set directory |
| `--iterations` | No | 3 | Refinement iterations |
| `--output-dir` | No | `outputs` | Output directory |

### `plot` (statistical plots)

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--data` | Yes | — | JSON file with raw data |
| `--intent` | Yes | — | Visual intent caption |
| `--reference-dir` | No | `data/reference_sets` | Reference set directory |
| `--iterations` | No | 3 | Refinement iterations |
| `--output-dir` | No | `outputs` | Output directory |

## Output

Results saved to `outputs/run_<timestamp>/`:

- `final_output.png` — final generated image
- `planning.json` — retrieved examples and description
- `iter_N.png` — image from each iteration
- `iter_N_details.json` — description and critic feedback per iteration

## Reference Sets

References are automatically downloaded from GitHub on first use. You can also manually set up references:

```bash
# Explicit setup to default cache (~/.paperbanana/reference_sets)
uv run scripts/paperbanana_lite.py setup

# Setup to a custom location
uv run scripts/paperbanana_lite.py setup --target-dir ./my-references
```

For custom reference sets, the `--reference-dir` must point to a directory containing an `index.json` with curated examples. Each example needs: `id`, `source_context`, `caption`, `image_path`. See `references/reference-set-format.md` for the schema.
