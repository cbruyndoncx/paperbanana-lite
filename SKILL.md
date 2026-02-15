---
name: paperbanana-lite
description: Generate publication-quality academic methodology diagrams and statistical plots from text descriptions using a multi-agent Gemini pipeline. Use when the user wants to create scientific figures, methodology diagrams, architecture diagrams, or statistical plots for academic papers. Triggers on requests like "generate a diagram", "create a methodology figure", "make a plot for my paper", or "illustrate this architecture".
---

# PaperBanana Lite

Single-file multi-agent pipeline for academic illustration generation using Google Gemini. Distills the full PaperBanana framework into one script.

## Quick Start

```bash
# Diagram generation
python scripts/paperbanana_lite.py generate \
  --input methodology.txt \
  --caption "Overview of our encoder-decoder architecture" \
  --reference-dir path/to/references

# Plot generation
python scripts/paperbanana_lite.py plot \
  --data results.json \
  --intent "Bar chart comparing model accuracy" \
  --reference-dir path/to/references
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

The `--reference-dir` must point to a directory containing an `index.json` with curated examples. Each example needs: `id`, `source_context`, `caption`, `image_path`. See `references/reference-set-format.md` for the schema.
